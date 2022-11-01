import asyncio
import json
import os
import sys
import time
from random import randint
from typing import Dict
import matplotlib.pyplot as plt
import requests

sys.path.append("./")
from common.fast_inference import FastInferenceInterface
from common.together_web3.together import TogetherWeb3 
from loguru import logger
from diffusers import StableDiffusionPipeline
import torch

class FastStableDiffusion(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        super().__init__(model_name, args if args is not None else {})
        self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, revision="fp16")
        self.pipe = self.pipe.to("cuda")

    def dispatch_request(self, args, env) -> Dict:
        print(args)
        print(env)

if __name__ == "__main__":
    fip = FastStableDiffusion(model_name="stable_diffusion", args={
        "coordinator":TogetherWeb3(coordinator="localhost", websocket_url="localhost:8093")
    })
    fip.start()