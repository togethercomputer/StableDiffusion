FROM 598726163780.dkr.ecr.us-west-2.amazonaws.com/node:latest AS node
# Forked from https://github.com/tridao/zoo/blob/0c43127363a6bcf54ff200f215654e24e344f9ae/Dockerfile
FROM nvcr.io/nvidia/pytorch:22.09-py3 as base

ENV HOST docker
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# https://serverfault.com/questions/683605/docker-container-time-timezone-will-not-reflect-changes
ENV TZ America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# git for installing dependencies
# tzdata to set time zone
# wget and unzip to download data
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    sudo \
    less \
    htop \
    git \
    tzdata \
    wget \
    tmux \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN mkdir -p /home/user/.together && chmod 777 /home/user
WORKDIR /home/user

# Disable pip cache: https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for
ENV PIP_NO_CACHE_DIR=1

# General packages that we don't care about the version
RUN pip install pytest matplotlib jupyter ipython ipdb gpustat scikit-learn spacy munch einops opt_einsum fvcore gsutil cmake pykeops together_web3 together_worker \
    && python -m spacy download en_core_web_sm

# Core packages
RUN pip install transformers==4.22.2 datasets==2.5.1

# # Install FlashAttention
RUN git clone https://github.com/togethercomputer/flash-attention \
    && cd flash-attention && pip install . \
    && cd .. && rm -rf flash-attention

# Set up diffusers
RUN git clone https://github.com/togethercomputer/diffusers \
    && cd diffusers && pip install -e .

COPY --from=node /usr/local/bin/together /usr/local/bin/
COPY app app
COPY serve.sh serve.sh
COPY local-cfg.yaml .together/cfg.yaml

CMD ./serve.sh
