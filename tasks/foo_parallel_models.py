import asyncio
import json
import os
import sys
import time
from random import randint
from typing import Dict
import matplotlib.pyplot as plt
import requests
import torch.cuda

import torch.multiprocessing as mp
from torch import distributed as dist
sys.path.append("./")
from common.fast_inference import FastInferenceInterface
from loguru import logger


def init_model(rank):
    dist.init_process_group("nccl", init_method='localhost:29500', rank=rank)
    torch.cuda.set_device(rank)
    fake_model = torch.full((10, 10), float(rank), device='cuda:' + str(rank))
    return fake_model


def start_foo_parallel_model(rank):
    fake_model = init_model(rank)
    raw_text= None
    while True:
        info = [raw_text]
        dist.broadcast_object_list(info)
        raw_text = info
        print(f"Rank<{rank}>, recv info: {raw_text}")
        current_output = fake_model.detach()
        dist.all_reduce(current_output)
        print(current_output)


class FooMultiModel(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        super().__init__(model_name, args)
        mp.set_start_method('spawn', force=True)
        # hard code 4 GPU so far
        self.process = []
        for rank in range(1,4):
            p = mp.Process(target=start_foo_parallel_model, args=(rank,))
            p.start()
            self.process.append(p)
        self.model = init_model(0)

    def infer(self, job_id, args):
        print(args)
        args = json.loads(args)
        print(args)
        start = time.time()
        info = args
        dist.broadcast_object_list(info)
        raw_text = info
        print(f"Rank<{0}>, recv info: {raw_text}")
        current_output = self.model.detach()
        dist.all_reduce(current_output)
        print(current_output)
        end = time.time()
        print(f"Rank<{0}> allreduce time {end-start}")
        # delete the file



if __name__ == "__main__":
    fip = FooMultiModel(model_name="FooMultiModel")
    fip.start()
    # to test...
    # fip.infer("random_id", {"arg_key":"arg_val"})