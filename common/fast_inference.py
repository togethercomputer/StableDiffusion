import asyncio
import os
from enum import Enum
from typing import Dict
from loguru import logger
from typing import Any
from .together_web3.computer import (
    Job,
    LanguageModelInferenceChoice,
    LanguageModelInferenceResult,
    MatchEvent,
    RequestTypeImageModelInference,
    RequestTypeLanguageModelInference,
    RequestTypeShutdown,
    ResultEnvelope,
)
from .together_web3.together import TogetherWeb3

class ServiceDomain(Enum):
    http = "http"
    together = "together"

class FastInferenceInterface:
    def __init__(self, model_name: str, args=None) -> None:
        self.model_name = model_name
        self.service_domain = args.get("service_domain", ServiceDomain.together)
        self.coordinator: TogetherWeb3 = args.get(
            "coordinator") if self.service_domain == ServiceDomain.together else None
        self.shutdown = False
        
    def infer(self, job_id, args) -> Dict:
        pass

    async def _run_together_server(self) -> None:
        logger.info("Start _run_together_server")
        self.coordinator._on_match_event.append(self.together_request)
        try:
            while not self.shutdown:
                await asyncio.sleep(1)
        except Exception as e:
            logger.exception(f'_run_together_server: {e}')
        self._shutdown()
    
    async def together_request(self, match_event: MatchEvent, raw_event: Dict[str, Any]) -> None:
        # logger.info("together_request %s", raw_event)
        request_json = [raw_event["match"]["service_bid"]["job"]]
        response = (await self.dispatch_request(request_json, match_event))
        await self.send_together_result(match_event, response.json[0])

    def start(self):
        loop = asyncio.get_event_loop()
        future = asyncio.Future()
        asyncio.ensure_future(self._run_together_server())
        loop.run_forever()

    def _shutdown(self) -> None:
        logger.info("Shutting down")

if __name__ == "__main__":
    fip = FastInferenceInterface(model_name="StableDiffusion")
    fip.start()
