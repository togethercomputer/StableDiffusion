import asyncio
import base64
import json
import os
import sys
import time
from io import BytesIO
from random import randint
from typing import Dict
import matplotlib.pyplot as plt
import requests

sys.path.append("./")
from common.fast_inference import FastInferenceInterface
from common.together_web3.computer import ImageModelInferenceChoice, RequestTypeImageModelInference
from common.together_web3.together import TogetherWeb3, TogetherClientOptions
from loguru import logger
from diffusers import StableDiffusionPipeline
import torch


class FastStableDiffusion(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        super().__init__(model_name, args if args is not None else {})
        self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, revision="fp16")
        self.pipe = self.pipe.to("cuda")

    def dispatch_request(self, args, env) -> Dict:
        prompt = args[0]["prompt"] 
        output = self.pipe(
            prompt[0] if isinstance(prompt, list) else prompt,
            #num_inference_steps=args[0].get("steps", 50),
        )
        print(output)

        choices = []
        for image in output.images:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
            choices.append(ImageModelInferenceChoice(img_str))
        return {
            "result_type": RequestTypeImageModelInference,
            "choices": choices,
        }

if __name__ == "__main__":
    coord_url = os.environ.get("COORD_URL", "127.0.0.1")
    coordinator = TogetherWeb3(
        TogetherClientOptions(),
        http_url=f"http://{coord_url}:8092",
        websocket_url=f"ws://{coord_url}:8093/websocket"
    )
    fip = FastStableDiffusion(model_name="stable_diffusion", args={
        "coordinator": coordinator,
    })
    fip.start()
