# Forked from https://github.com/tridao/zoo/blob/0c43127363a6bcf54ff200f215654e24e344f9ae/Dockerfile
FROM nvcr.io/nvidia/pytorch:22.09-py3 as base

ENV HOST docker
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# https://serverfault.com/questions/683605/docker-container-time-timezone-will-not-reflect-changes
ENV TZ America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV DEBIAN_FRONTEND=noninteractive

# git for installing dependencies
# tzdata to set time zone
# wget and unzip to download data
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    coinor-cbc \
    curl \
    ca-certificates \
    sudo \
    less \
    htop \
    git \
    python3-pip \
    python3-dev \
    tzdata \
    wget \
    tmux \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://together-distro-packages.s3.us-west-2.amazonaws.com/linux/x86_64/bin/together-node-latest -O /usr/local/bin/together-node && \
    chmod +x /usr/local/bin/together-node

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN mkdir -p /home/user/.together && chmod 777 /home/user
WORKDIR /home/user

# Disable pip cache: https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for
ENV PIP_NO_CACHE_DIR=1

# General packages that we don't care about the version
RUN pip install pytest matplotlib jupyter ipython ipdb gpustat spacy munch opt_einsum fvcore gsutil cmake pykeops together_web3 together_worker diffusers[torch] \
    && python -m spacy download en_core_web_sm

# Core packages
RUN pip install torch==2.0.1 transformers==4.31.0 datasets==2.5.1 accelerate==0.21.0 \
    && torchvision==0.15.2 tokenizers==0.13.3 sentencepiece==0.1.99 peft==0.4.0 \
    && bitsandbytes==0.41.0 einops==0.6.1 timm==0.6.13

# # Install FlashAttention
RUN pip3 install --no-cache-dir \
    'flash_attn @ git+https://github.com/Dao-AILab/flash-attention.git@v2.2.3' \
    xformers

# Install non-simd Pillow
RUN pip install --upgrade Pillow

# Fix undefined symbol bug
RUN pip uninstall transformer-engine -y

COPY local-cfg.yaml /home/user/cfg.yaml
COPY app app
COPY serve.sh serve.sh

CMD ./serve.sh
