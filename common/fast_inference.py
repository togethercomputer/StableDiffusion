import asyncio
import os
from typing import Dict

import nats
from nats.errors import ConnectionClosedError, NoServersError, TimeoutError


class FastInferenceInterface:
    def __init__(self, model_name: str, args=None) -> None:
        self.model_name = model_name

    def infer(self, job_id, args) -> Dict:
        pass

    async def on_message(self, msg):
        instruction = msg.data.decode("utf-8")
        job_id, job_payload = str(instruction).split(":", 1)
        self.infer(job_id, job_payload)
        
    def on_error(self, ws, msg):
        print(msg)

    def on_open(self, ws):
        ws.send(f"JOIN:{self.model_name}")

    def start(self):
        nats_url = os.environ.get("NATS_URL", "localhost:8092/my_coord")
        async def listen():
            nc = await nats.connect(f"nats://{nats_url}")
            sub = await nc.subscribe(subject='StableDiffusion', queue="StableDiffusion", cb=self.on_message)
        loop = asyncio.get_event_loop()
        future = asyncio.Future()
        asyncio.ensure_future(listen())
        loop.run_forever()

if __name__ == "__main__":
    fip = FastInferenceInterface(model_name="StableDiffusion")
    fip.start()
