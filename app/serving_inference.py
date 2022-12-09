import os
import base64
import logging
from io import BytesIO
from typing import Dict
import torch

from diffusers import StableDiffusionPipeline
from together_worker.fast_inference import FastInferenceInterface
from together_web3.together import TogetherWeb3, TogetherClientOptions
from together_web3.computer import ImageModelInferenceChoice, RequestTypeImageModelInference

class FastStableDiffusion(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        args = args if args is not None else {}
        super().__init__(model_name, args)
        self.pipeline_name = args.get("pipeline", "runwayml/stable-diffusion-v1-5")
        self.default_size = 512
        if self.pipeline_name == "stabilityai/stable-diffusion-2-1":
            self.default_size = 768
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.pipeline_name,
            torch_dtype=torch.float16,
            revision="fp16",
            use_auth_token=args.get("auth_token"),
        )
        self.format = args.get("format", "JPEG")
        self.device = args.get("device", "cuda")
        self.pipe = self.pipe.to(self.device)

    def dispatch_request(self, args, env) -> Dict:
        try:
            prompt = args[0]["prompt"]
            seed = args[0].get("seed")
            generator = torch.Generator(self.device).manual_seed(seed) if seed else None
            
            output = self.pipe(
                prompt if isinstance(prompt, list) else [prompt],
                generator=generator,
                height=int(args[0].get("height", self.default_size)),
                width=int(args[0].get("width", self.default_size)),
                num_images_per_prompt=int(args[0].get("n", 1)),
                num_inference_steps=int(args[0].get("steps", 50)),
                guidance_scale=int(args[0].get("guidance_scale", 7.5)),
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
    pipeline_name = os.environ.get("PIPELINE", "runwayml/stable-diffusion-v1-5")
    url_friendly_name = pipeline_name.replace("/", "-")
    coord_url = os.environ.get("COORD_URL", "127.0.0.1")
    coordinator = TogetherWeb3(
        TogetherClientOptions(reconnect=True),
        http_url=os.environ.get("COORD_HTTP_URL", f"http://{coord_url}:8092"),
        websocket_url=os.environ.get("COORD_WS_URL", f"ws://{coord_url}:8093/websocket"),
    )
    fip = FastStableDiffusion(model_name=url_friendly_name, args={
        "auth_token": os.environ.get("AUTH_TOKEN"),
        "coordinator": coordinator,
        "device": os.environ.get("DEVICE", "cuda"),
        "format": os.environ.get("FORMAT", "JPEG"),
        "gpu_num": 1 if torch.cuda.is_available() else 0,
        "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "group_name": os.environ.get("GROUP", "group1"),
        "worker_name": os.environ.get("WORKER", "worker1"),
        "pipeline": os.environ.get("PIPELINE", "runwayml/stable-diffusion-v1-5"),
    })
    fip.start()
