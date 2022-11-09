import time
import torch
from diffusers import StableDiffusionPipeline
import functools
import argparse

# torch disable grad
torch.set_grad_enabled(False)

torch.manual_seed(1231)
torch.cuda.manual_seed(1231)

prompt = "a photo of an astronaut riding a horse on mars"

# cudnn benchmarking
torch.backends.cudnn.benchmark = True

# make sure you're logged in with `huggingface-cli login`
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
).to("cuda")

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()
batch_size = args.batch_size

# warmup
with torch.inference_mode():
    image = pipe([prompt] * batch_size, num_inference_steps=5).images[0]

for _ in range(3):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.inference_mode():
        image = pipe([prompt] * batch_size, num_inference_steps=50).images[0]
    torch.cuda.synchronize()
    print(f"Pipeline inference for {batch_size} images took {time.time() - start_time:.2f} seconds")