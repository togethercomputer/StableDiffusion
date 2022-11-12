from typing import Any, Awaitable, Callable, Dict, List, Optional, Union, cast

import asyncio
import dataclasses
import json
import logging
import random
from asyncio import Future, Task
from dataclasses import asdict, dataclass, fields
from enum import Enum

from dacite import from_dict
from web3 import HTTPProvider, Web3
from websockets.client import connect

from .accounts import TogetherAccounts, TogetherAccountsProtocol
from .computer import (
    AddressNil,
    BlockHeader,
    EventEnvelope,
    EventTypeMatch,
    EventTypeNewBlock,
    EventTypeResult,
    ImageModelInferenceRequest,
    ImageModelInferenceResult,
    Job,
    LanguageModelInferenceRequest,
    LanguageModelInferenceResult,
    MatchEvent,
    NewBlockEvent,
    OfferEnvelope,
    OfferTypeServiceBid,
    RequestTypeImageModelInference,
    RequestTypeLanguageModelInference,
    ResourceTypeService,
    ResultEnvelope,
    ResultEvent,
    Service,
    ServiceBid,
    ServicePrice,
    TogetherComputer,
    TogetherComputerProtocol,
)
from .coordinator import TogetherCoordinator, TogetherCoordinatorProtocol

logger = logging.getLogger(__name__)


@dataclass
class ResolveOptions:
    #: Callback for MatchEvent with associated offer_id.
    match_callback: Optional[Callable[[Dict[str, Any]], None]]


@dataclass
class TogetherClientOptions:
    maintainConnection: Optional[bool] = None


def asdict_filter_none(x):
    return asdict(x, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})


def dataclass_replace_none_with_default_values(data_class, data):
    return from_dict(data_class=data_class, data=asdict_filter_none(data))


def websocket_url_from_http_url(http_url: str) -> str:
    separator = '/' if len(http_url) == 0 or http_url[-1] != '/' else ''
    if http_url.startswith('http://'):
        link = http_url[7:]
        return f"ws://{link}{separator}websocket"
    if http_url.startswith('https://'):
        link = http_url[8:]
        return f"wss://{link}{separator}websocket"
    else:
        return f"ws://{http_url}{separator}websocket"


class TogetherWeb3:
    """The TogetherWeb3 class is an umbrella to house all Together Computer related modules."""
    accounts: TogetherAccountsProtocol
    coordinator: TogetherCoordinatorProtocol
    together: TogetherComputerProtocol
    _tip_block: Optional[BlockHeader] = None
    _subscription: "Optional[Task[None]]" = None
    _subscription_id: Optional[str] = None
    _on_subscription_id: "List[Future[str]]" = []
    _on_new_block_header: "List[Future[BlockHeader]]" = []
    _on_match_event: "List[Callable[[MatchEvent, Dict[str, Any]], Union[None, Awaitable[None]]]]" = []
    _on_match_for_bid: "Dict[str, List[Callable[[Dict[str, Any]], None]]]" = {}
    _on_result_for_bid: "Dict[str, List[Future[Dict[str, Any]]]]" = {}

    def __init__(
        self,
        options: TogetherClientOptions = TogetherClientOptions(),
        http_url: str = "https://computer.together.xyz",
        websocket_url: Optional[str] = None,
        **kwargs: Any
    ):
        self.http_url = http_url
        self.websocket_url = websocket_url if websocket_url else websocket_url_from_http_url(
            http_url)
        self.options = options
        self.web3 = Web3(
            provider=HTTPProvider(self.http_url),
            modules={
                "accounts": TogetherAccounts,
                "coordinator": TogetherCoordinator,
                "together": TogetherComputer,
            }
        )
        self.accounts = cast(Any, self.web3).accounts
        self.coordinator = cast(Any, self.web3).coordinator
        self.together = cast(Any, self.web3).together

    def subscribe_events(self, rpc_namespace: str = "together") -> None:
        """Opens WebSocket connection to the Together Computer."""
        if not self._subscription:
            self._subscription = asyncio.create_task(
                self._handle_events(f"{rpc_namespace}_subscribe", "events", self._handle_event))

    async def _handle_events(self, method: str, event: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        while True:
            ws = await connect(self.websocket_url)
            await ws.send(f'{{"jsonrpc": "2.0", "id": 1, "method": "{method}", "params": ["{event}"]}}')
            message = await ws.recv()
            subscription_response = json.loads(message)
            self._subscription_id = subscription_response["result"]
            resolve_subscription_id = self._on_subscription_id
            self._on_subscription_id = []
            for future_subscription_id in resolve_subscription_id:
                future_subscription_id.set_result(
                    self._subscription_id if self._subscription_id else "")
            try:
                while True:
                    message = await asyncio.wait_for(ws.recv(), 1073741824)
                    response = json.loads(message)
                    result = response["params"]["result"]
                    # logger.info("_handle_events: %s", result)
                    await handler(result)
            except asyncio.CancelledError:
                break
            except BaseException:
                logging.exception("handle_events")
            finally:
                await ws.close()
            self._subscription_id = None
            await asyncio.sleep(2)

    async def _handle_event(self, result: Dict[str, Any]) -> None:
        update = from_dict(data_class=EventEnvelope, data=result)
        if update.events[0].event_type == EventTypeNewBlock:
            await self._handle_new_block_event(
                from_dict(
                    data_class=NewBlockEvent,
                    data=result["events"][0]))
        elif update.events[0].event_type == EventTypeMatch:
            await self._handle_match_event(
                from_dict(
                    data_class=MatchEvent,
                    data=result["events"][0]),
                result["events"][0])
        elif update.events[0].event_type == EventTypeResult:
            await self._handle_result_event(
                from_dict(
                    data_class=ResultEvent,
                    data=result["events"][0]),
                result["events"][0])
        else:
            logger.error(f"unknown event_type: {update.events[0].event_type}")

    async def _handle_new_block_event(self, new_block: NewBlockEvent) -> None:
        """Resolve Futures listening for the next block."""
        self._tip_block = new_block
        resolve_new_block_header = self._on_new_block_header
        self._on_new_block_header = []
        for future_new_block_header in resolve_new_block_header:
            future_new_block_header.set_result(self._tip_block)

    async def _handle_match_event(self, match_event: MatchEvent, raw_event: Dict[str, Any]) -> None:
        """Dispatch callbacks listening for MatchEvent with some offer_id."""
        bid_offer_id = match_event.match.bid_offer_id
        if bid_offer_id in self._on_match_for_bid:
            resolve_match_bid = self._on_match_for_bid[bid_offer_id]
            for callback_match_bid in resolve_match_bid:
                callback_match_bid(raw_event)
            del self._on_match_for_bid[bid_offer_id]
        for callback_match in self._on_match_event:
            result = callback_match(match_event, raw_event)
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                await result

    async def _handle_result_event(self, result_event: ResultEvent,
                                   raw_event: Dict[str, Any]) -> None:
        """Resolve Futures listening for ResultEvent with some offer_id."""
        bid_offer_id = result_event.result.bid_offer_id
        if bid_offer_id in self._on_result_for_bid:
            resolve_result = self._on_result_for_bid[bid_offer_id]
            for future_result in resolve_result:
                future_result.set_result(raw_event)
            del self._on_result_for_bid[bid_offer_id]

    async def close(self) -> None:
        """Shutdown any together-web3 providers."""
        if self._subscription:
            self._subscription.cancel()
            await self._subscription
            self._subscription = None

    async def get_subscription_id(self, rpc_namespace: str = "together") -> str:
        """Returns subscription ID where asynchronous results can be sent."""
        future: Future[str] = Future()
        if self._subscription_id:
            future.set_result(self._subscription_id)
            return await future
        self._on_subscription_id.append(future)
        self.subscribe_events(rpc_namespace)
        return await future

    async def get_tip_block(self) -> BlockHeader:
        """Returns the BlockHeader from the latest block in the chain."""
        future: Future[BlockHeader] = Future()
        if self._tip_block:
            future.set_result(self._tip_block)
            return await future
        self._on_new_block_header.append(future)
        self.subscribe_events()
        return await future

    async def update_offer(self, update: OfferEnvelope) -> str:
        """Publishes an Offer to the network."""
        # logger.info("update_offer: %s", asdict(update))
        offer_id = self.together.updateOffer(asdict(update), self._subscription_id)
        return offer_id

    async def update_result(self, result: ResultEnvelope) -> str:
        result_id = self.coordinator.updateResult(asdict(result), self._subscription_id)
        return result_id

    async def resolve_offer(
        self,
        offer: OfferEnvelope,
        options: Optional[ResolveOptions] = None
    ) -> Dict[str, Any]:
        """Publishes and manages an Offer's resolution."""

        # Send the offer to market.
        offer_id = await self.update_offer(offer)
        # logger.info("resolve_offer, offer_id: %s", offer_id)

        # Listen for MatchEvent with our offer_id.
        if options and options.match_callback:
            if offer_id in self._on_match_for_bid:
                waiting_match_for_bid = self._on_match_for_bid[offer_id]
            else:
                waiting_match_for_bid = []
                self._on_match_for_bid[offer_id] = waiting_match_for_bid
            waiting_match_for_bid.append(options.match_callback)

        # Listen for ResultEvent with our offer_id.
        future: Future[Dict[str, Any]] = Future()
        if offer_id in self._on_result_for_bid:
            waiting_result_for_bid = self._on_result_for_bid[offer_id]
        else:
            waiting_result_for_bid = []
            self._on_result_for_bid[offer_id] = waiting_result_for_bid
        waiting_result_for_bid.append(future)
        return await future

    async def resolve_job(
        self,
        service: Service,
        job: Job,
        options: Optional[ResolveOptions] = None,
    ) -> Dict[str, Any]:
        """Publishes and manages a Job's resolution."""

        tip_block = await self.get_tip_block()
        service_bid = from_dict(
            data_class=ServiceBid,
            data={
                "offer_type": OfferTypeServiceBid,
                "block_height": tip_block.block_height,
                "block_nonce": tip_block.block_nonce,
                "nonce": random.getrandbits(32),
                "address": AddressNil,
                "market_address": tip_block.market_address,
                "service": service,
                "max_service_price": from_dict(
                    data_class=ServicePrice,
                    data={"base_price": 0},
                ),
                "job": job,
            }
        )
        offer_envelope = from_dict(
            data_class=OfferEnvelope,
            data={"offer": service_bid}
        )
        return await self.resolve_offer(offer_envelope, options)

    async def resolve_inference(
        self,
        request: Union[ImageModelInferenceRequest, LanguageModelInferenceRequest],
        options: Optional[ResolveOptions] = None,
    ) -> Dict[str, Any]:
        service = from_dict(
            data_class=Service,
            data={
                "resource_type": ResourceTypeService,
                "service_name": request.model,
            },
        )
        result_json = await self.resolve_job(service, request, options)
        return result_json

    async def language_model_inference(
        self,
        request: LanguageModelInferenceRequest,
        options: Optional[ResolveOptions] = None,
    ) -> LanguageModelInferenceResult:
        """Publishes and manages a LanguageModelInferenceRequest and the resulting LanguageModelInferenceResult."""
        result_json = await self.resolve_inference(request, options)
        return from_dict(data_class=LanguageModelInferenceResult, data=result_json["result"])

    async def image_model_inference(
        self,
        request: ImageModelInferenceRequest,
        options: Optional[ResolveOptions] = None,
    ) -> ImageModelInferenceResult:
        result_json = await self.resolve_inference(request, options)
        return from_dict(data_class=ImageModelInferenceResult, data=result_json["result"]['data'])