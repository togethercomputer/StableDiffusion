# Stable Diffusion with FlashAttention

To test:
```
python test/test.py
```

Expected performance on A100: 1.5-1.6s.


To bring up a standalone node:

```
mkdir .together
docker run --rm --gpus device=0 \
  -e AUTH_TOKEN=hf_XXX \
  -v $PWD/.together:/home/user/.together \
  -it togethercomputer/stablediffusion /usr/local/bin/together-node start \
    --config /home/user/cfg.yaml --color --worker.model runwayml/stable-diffusion-v1-5 \
    --worker.service StableDiffusion
```

To support image inputs add:
```
  --worker.options="input=text,image"
```

To run on all available GPUs:

```
docker run --rm --gpus all \
  -e AUTH_TOKEN=hf_XXX \
  -e NUM_WORKERS=auto \
  -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  -v $PWD/.together:/home/user/.together \
  -it togethercomputer/stablediffusion /usr/local/bin/together-node start \
    --config /home/user/cfg.yaml --color --worker.model runwayml/stable-diffusion-v1-5 \
    --worker.service StableDiffusion
```
