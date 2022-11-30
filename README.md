# Stable Diffusion with FlashAttention

To test:
```
python test/test.py
```

Expected performance on A100: 1.5-1.6s.


To bring up a standalone node:
```
mkdir models
mkdir .together
docker run --rm --gpus all \
  -e AUTH_TOKEN=hf_XXX \
  -e NUM_WORKERS=auto \
  -v $PWD/models:/together/together_models \
  -v $PWD/.together:/home/user/.together/together \
  -it togethercomputer/stablediffusion /usr/local/bin/together start
```
