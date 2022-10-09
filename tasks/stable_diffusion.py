import asyncio
import json
import os
import sys
import time
from random import randint
from typing import Dict
import matplotlib.pyplot as plt
import requests
from tensorflow import keras

sys.path.append("./")
from common.fast_inference import FastInferenceInterface
from loguru import logger
from diffusers import StableDiffusionPipeline
import torch


class FastStableDiffusion():
    def __init__(self, model_name: str, args=None) -> None:
        super().__init__(model_name, args)
        self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_type=torch.float16, revision="fp16")
        self.pipe = self.pipe.to("cuda")

    def infer(self, job_id, args) -> Dict:
        print(args)
        args = json.loads(args)
        print(args)
        start = time.time()
        images = []
        for i in range(args["num_returns"]):
            image = self.pipe(args['input'][0]).images
            images.extend(image)
        end = time.time()
        # save images to file
        filenames = []
        for i, image in enumerate(images):
            fileid = randint(0, 100000)
            plt.imsave(f"image_{fileid}.png", image)
            # upload images to s3
            with open(f"image_{fileid}.png", "rb") as fp:
                files = {"file": fp}
                res = requests.post("https://planetd.shift.ml/file", files=files).json()
                filenames.append(res["filename"])
            os.remove(f"image_{fileid}.png")
        # delete the file
        print("sending requests to global")
        # write results back
        coord_url = os.environ.get("COORDINATOR_URL", "localhost:8092/my_coord")
        worker_name = os.environ.get("WORKER_NAME", "planetv2")
        requests.patch(
            f"http://{coord_url}/api/v1/g/jobs/{job_id}",
            json={
                "status": "finished",
                "returned_payload": {"output": [filenames]},
                "source": "dalle",
                "type": "general",
                "processed_by": worker_name,
            },
        )


if __name__ == "__main__":
    fip = FastStableDiffusion(model_name="stable_diffusion")
    fip.infer("job", {"args":"ss"})
