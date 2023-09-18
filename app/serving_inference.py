import os
import base64
import logging
from io import BytesIO
from PIL import Image
from typing import Dict
import torch

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
)
from diffusers.utils import load_image
from together_worker.fast_inference import FastInferenceInterface
from together_web3.together import TogetherWeb3, TogetherClientOptions
from together_web3.computer import (
    ImageModelInferenceChoice,
    parse_tags,
    RequestTypeImageModelInference,
)


class FastStableDiffusion(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        args = args if args is not None else {}
        super().__init__(model_name, args)

        self.format = args.get("format", "JPEG")
        self.device = args.get("device", "cuda")

        model_revision = os.environ.get("MODEL_REVISION", "fp16")
        if not model_revision or model_revision == "none":
            model_revision = None

        self.model = os.environ.get("MODEL", "runwayml/stable-diffusion-v1-5")
        print("init MODEL", self.model)

        if self.model == "stabilityai/stable-diffusion-xl-base-1.0":
            self.pipe_text2image = StableDiffusionXLPipeline.from_pretrained(
                self.model,
                torch_dtype=torch.float16,
                revision=model_revision,
                use_auth_token=args.get("auth_token"),
                use_safetensors=True,
                variant="fp16",
                device_map="auto" if self.device == "cuda" else self.device,
            )
            self.pipe_text2image.enable_xformers_memory_efficient_attention()
        else:
            self.pipe_text2image = StableDiffusionPipeline.from_pretrained(
                self.model,
                torch_dtype=torch.float16,
                revision=model_revision,
                use_auth_token=args.get("auth_token"),
                device_map="auto" if self.device == "cuda" else self.device,
            )
            self.pipe_text2image.enable_xformers_memory_efficient_attention()

        self.options = parse_tags(os.environ.get("MODEL_OPTIONS"))
        self.inputs = self.options.get("input", "").split(",")

        if "image" in self.inputs:
            # use from_pipe to avoid consuming additional memory when loading a checkpoint
            self.pipe_image2image = AutoPipelineForImage2Image.from_pipe(
                self.pipe_text2image
            ).to(self.device)
            self.pipe_image2image.enable_xformers_memory_efficient_attention()

        # Commenting out for now
        # TODO: add support for inpainting
        # if "inpainting" in self.inputs:
        #    self.pipe_inpainting = AutoPipelineForInpainting.from_pipe(
        #        self.pipe_text2image
        #    ).to("cuda")

    def dispatch_request(self, args, env) -> Dict:
        try:
            print("INVOKING:", self.model)
            print("ARGS:", args[0])
            prompt = args[0]["prompt"]
            negative_prompt = args[0].get("negative_prompt", None)
            if negative_prompt is not None:
                negative_prompt = (
                    negative_prompt
                    if isinstance(negative_prompt, list)
                    else [negative_prompt]
                )
            seed = args[0].get("seed")
            generator = torch.Generator(self.device).manual_seed(seed) if seed else None
            image_input = args[0].get("image_base64")
            if image_input:
                if not self.pipe_image2image:
                    raise Exception("Image prompts not supported")

                init_image = load_image(image_input).convert("RGB")
                init_image.thumbnail((768, 768))

                output = self.pipe_image2image(
                    prompt if isinstance(prompt, list) else [prompt],
                    negative_prompt=negative_prompt,
                    image=init_image,
                    generator=generator,
                    num_images_per_prompt=args[0].get("n", 1),
                    num_inference_steps=args[0].get("steps", 50),
                    guidance_scale=args[0].get("guidance_scale", 7.5),
                    strength=args[0].get("strength", 0.75),  # must be between 0 and 1
                )
            else:
                output = self.pipe_text2image(
                    prompt if isinstance(prompt, list) else [prompt],
                    negative_prompt=negative_prompt,
                    generator=generator,
                    height=args[0].get("height", 512),
                    width=args[0].get("width", 512),
                    num_images_per_prompt=args[0].get("n", 1),
                    num_inference_steps=args[0].get("steps", 50),
                    guidance_scale=args[0].get("guidance_scale", 7.5),
                )
            choices = []
            for image in output.images:
                buffered = BytesIO()
                image.save(buffered, format=args[0].get("format", self.format))
                img_str = base64.b64encode(buffered.getvalue()).decode("ascii")
                choices.append(ImageModelInferenceChoice(img_str))
            return {
                "result_type": RequestTypeImageModelInference,
                "choices": choices,
            }
        except Exception as e:
            logging.exception(e)
            return {
                "result_type": "error",
                "value": str(e),
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    coord_url = os.environ.get("COORD_URL", "localhost")
    coordinator = TogetherWeb3(
        TogetherClientOptions(reconnect=True),
        http_url=os.environ.get("COORD_HTTP_URL", f"http://{coord_url}:8092"),
        websocket_url=os.environ.get(
            "COORD_WS_URL", f"ws://{coord_url}:8093/websocket"
        ),
    )
    fip = FastStableDiffusion(
        model_name=os.environ.get("SERVICE", "StableDiffusion"),
        args={
            "auth_token": os.environ.get("AUTH_TOKEN"),
            "coordinator": coordinator,
            "device": os.environ.get("DEVICE", "cuda"),
            "format": os.environ.get("FORMAT", "JPEG"),
            "gpu_num": 1 if torch.cuda.is_available() else 0,
            "gpu_type": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "gpu_mem": torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else None,
            "group_name": os.environ.get("GROUP", "group1"),
            "http_host": os.environ.get("HTTP_HOST"),
            "http_port": int(os.environ.get("HTTP_PORT", "5001")),
            "service_domain": os.environ.get("SERVICE_DOMAIN", "together"),
            "worker_name": os.environ.get("WORKER", "worker1"),
        },
    )
    fip.start()
