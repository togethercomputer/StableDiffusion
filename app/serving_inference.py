import os
import base64
import logging
from io import BytesIO
from PIL import Image
from typing import Dict
import torch

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionXLPipeline
from together_worker.fast_inference import FastInferenceInterface
from together_web3.together import TogetherWeb3, TogetherClientOptions
from together_web3.computer import ImageModelInferenceChoice, parse_tags, RequestTypeImageModelInference


class FastStableDiffusion(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        args = args if args is not None else {}
        super().__init__(model_name, args)
        model_revision = os.environ.get("MODEL_REVISION", "fp16")
        if not model_revision or model_revision == "none":
            model_revision = None
        model = os.environ.get("MODEL", "runwayml/stable-diffusion-v1-5")
        if(model == "stabilityai/stable-diffusion-xl-base-1.0"):
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model,
                torch_dtype=torch.float32,
                revision=model_revision,
                use_auth_token=args.get("auth_token"),
                use_safetensors=True,
                variant="fp16"
            )
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model,
                torch_dtype=torch.float16,
                revision=model_revision,
                use_auth_token=args.get("auth_token"),
            )
        self.format = args.get("format", "JPEG")
        self.device = args.get("device", "cuda")
        self.pipe = self.pipe.to(self.device)
        self.options = parse_tags(os.environ.get("MODEL_OPTIONS"))
        self.inputs = self.options.get("input", "").split(",")
        if "image" in self.inputs:
            self.image_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                os.environ.get("MODEL", "runwayml/stable-diffusion-v1-5"),
                torch_dtype=torch.float16,
                revision=model_revision,
                use_auth_token=args.get("auth_token"),
            ).to(self.device)


    def dispatch_request(self, args, env) -> Dict:
        try:
            prompt = args[0]["prompt"]
            seed = args[0].get("seed")
            generator = torch.Generator(self.device).manual_seed(seed) if seed else None
            image_input = args[0].get("image_base64")
            if image_input:
                if not self.image_pipe:
                    raise Exception("Image prompts not supported")
                init_image = Image.open(BytesIO(base64.b64decode(image_input))).convert("RGB")
                init_image.thumbnail((768, 768))
                output = self.image_pipe(
                    prompt if isinstance(prompt, list) else [prompt],
                    init_image=init_image,
                    generator=generator,
                    height=args[0].get("height", 512),
                    width=args[0].get("width", 512),
                    num_images_per_prompt=args[0].get("n", 1),
                    num_inference_steps=args[0].get("steps", 50),
                    guidance_scale=args[0].get("guidance_scale", 7.5),
                )
            else:
                output = self.pipe(
                    prompt if isinstance(prompt, list) else [prompt],
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
                img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
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
        websocket_url=os.environ.get("COORD_WS_URL", f"ws://{coord_url}:8093/websocket"),
    )
    fip = FastStableDiffusion(model_name=os.environ.get("SERVICE", "StableDiffusion"), args={
        "auth_token": os.environ.get("AUTH_TOKEN"),
        "coordinator": coordinator,
        "device": os.environ.get("DEVICE", "cuda"),
        "format": os.environ.get("FORMAT", "JPEG"),
        "gpu_num": 1 if torch.cuda.is_available() else 0,
        "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_mem": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "group_name": os.environ.get("GROUP", "group1"),
        "http_host": os.environ.get("HTTP_HOST"),
        "http_port": int(os.environ.get("HTTP_PORT", "5001")),
        "service_domain": os.environ.get("SERVICE_DOMAIN", "together"),
        "worker_name": os.environ.get("WORKER", "worker1"),
    })
    fip.start()
