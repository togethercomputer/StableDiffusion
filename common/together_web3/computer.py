from typing import Any, Callable, Dict, List, Optional, cast
from typing_extensions import Protocol

from dataclasses import dataclass

from web3.method import Method, default_root_munger
from web3.module import Module
from web3.types import RPCEndpoint

# ID is an offers's unique identifier.
ID = str  # Keccak-256 hash.

# Address represents a unique account.
Address = str  # Keccak-256 hash of the public key.

# Signature is an offer's signature.
Signature = str

AddressNil = "0x0000000000000000000000000000000000000000"
EventTypeMatch = "match"
EventTypeNewBlock = "new-block"
EventTypeResult = "result"
OfferTypeInstanceAsk = "instance-ask"
OfferTypeInstanceBid = "instance-bid"
OfferTypeRemove = "remove"
OfferTypeServiceAsk = "service-ask"
OfferTypeServiceBid = "service-bid"
RequestTypeImageModelInference = "image-model-inference"
RequestTypeLanguageModelInference = "language-model-inference"
RequestTypeShutdown = "shutdown"
RequestTypeStatus = "status"
ResourceTypeInstance = "instance"
ResourceTypeService = "service"


@dataclass
class Job:
    """Base class for work fulfilled by a Service."""
    request_type: str


@dataclass
class Event:
    """Base class for information emitted by the network."""
    event_type: str


############################################################
# Resources

@dataclass
class Resource:
    """Base class for resource commodity."""
    resource_type: str
    tags: Optional[Dict[str, str]]


@dataclass
class Storage:
    type: str
    capacity: int


@dataclass
class Container:
    command: Optional[List[str]]
    env: Optional[List[str]]
    image: str


@dataclass
class Instance(Resource):
    arch: Optional[str] = None
    os: Optional[str] = None
    cpu_num: Optional[int] = None
    cpu_type: Optional[str] = None
    gpu_num: Optional[int] = None
    gpu_type: Optional[str] = None
    gpu_tflops: Optional[int] = None
    gpu_memory: Optional[int] = None
    memory: Optional[int] = None
    network_up: Optional[int] = None
    network_down: Optional[int] = None
    storage: Optional[List[Storage]] = None


@dataclass
class Service(Resource):
    service_name: str


############################################################
# Resource prices

@dataclass
class ResourcePrice:
    base_price: int


@dataclass
class InstancePrice(ResourcePrice):
    network_up_price: Optional[int]
    network_down_price: Optional[int]


@dataclass
class ServicePrice(ResourcePrice):
    pass


############################################################
# Block format

@dataclass
class BlockHeader:
    block_previous: str
    block_height: int
    block_nonce: int
    market_address: Address


@dataclass
class Block(BlockHeader):
    transactions: List[Event]
    signature: Optional[Signature]


############################################################
# Offers

@dataclass
class Offer:
    offer_type: str
    offer_previous: Optional[str]
    block_height: int
    block_nonce: int
    address: Address
    market_address: Address
    memo: Optional[str]


@dataclass
class ServiceAsk(Offer):
    service: Service
    service_price: ServicePrice
    available_instance: Optional[Instance]
    total_instance: Optional[Instance]
    load_delay: Optional[int]
    high_water_mark: Optional[int]
    revenue_share: Optional[Dict[str, int]]


@dataclass
class ServiceBid(Offer):
    service: Service
    max_service_price: ServicePrice
    job: Job


@dataclass
class InstanceAsk(Offer):
    instance: Instance
    instance_price: InstancePrice
    num_instances: int
    container_cached: Optional[List[str]]


@dataclass
class InstanceBid(Offer):
    instance: Instance
    max_instance_price: InstancePrice
    num_instances: int
    container: Container
    supply_service: Optional[ServiceAsk]


############################################################
# Matches

@dataclass
class Match:
    match_type: str
    ask_offer_id: str
    bid_offer_id: str


@dataclass
class MatchServiceOffer(Match):
    supply_service: ServiceAsk
    demand_service: ServiceBid


@dataclass
class MatchInstanceOffer(Match):
    supply_instance: InstanceAsk
    demand_instance: InstanceBid


############################################################
# Jobs

@dataclass
class LanguageModelInferenceRequest(Job):
    """
    Based on https://beta.openai.com/docs/api-reference/completions
    Note: not all fields are supported right now (e.g., stream).
    """
    request_type: str = RequestTypeLanguageModelInference
    model: str = ""
    prompt: str = ""

    #: Maximum number of tokens to generate.
    max_tokens: Optional[int] = None

    #: Annealing temperature.
    temperature: Optional[float] = None

    #: Fraction of probability mass to keep (in top-p sampling).
    top_p: Optional[float] = None

    #: Number of samples to generate.
    n: Optional[int] = None

    #: Number of tokens to show logprobs.
    logprobs: Optional[int] = None

    #: Include the input as part of the output (e.g., for language modeling).
    echo: Optional[bool] = None

    #: Produce This many candidates per token.
    best_of: Optional[int] = None

    #: Stop when any of these strings are generated.
    stop: Optional[List[str]] = None

    stream_tokens: Optional[bool] = None


@dataclass
class ImageModelInferenceRequest(Job):
    """
    Request for image generation models (e.g., Stable Diffusion).
    API roughly based on https://github.com/CompVis/stable-diffusion
    TODO: add other parameters
    """
    request_type: str = RequestTypeImageModelInference
    model: str = ""
    prompt: str = ""

    #: How wide of an image to generate.
    width: Optional[int] = 512

    #: How tall of an image to generate.
    height: Optional[int] = 512

    #: Input image.
    image_base64: Optional[str] = ""

    downsampling_factor: Optional[float] = 0

    # Number of samples to draw.
    n: Optional[int] = 1


############################################################
# Results

@dataclass
class Result:
    ask_offer_id: str
    bid_offer_id: str
    match_id: str
    result_type: str
    partial: Optional[bool]


@dataclass
class LanguageModelInferenceChoice:
    # There are more fields here that aren't specified.
    # See the OpenAI API.
    text: str


@dataclass
class LanguageModelInferenceResult(Result):
    choices: List[LanguageModelInferenceChoice]


@dataclass
class ImageModelInferenceChoice:
    image_base64: str


@dataclass
class ImageModelInferenceResult(Result):
    choices: List[ImageModelInferenceChoice]


############################################################
# Envelopes

@dataclass
class Envelope:
    signature: Optional[Signature]


@dataclass
class OfferEnvelope(Envelope):
    offer: Offer
    offer_id: Optional[str]


@dataclass
class MatchEnvelope(Envelope):
    match: Match
    match_id: Optional[str]


@dataclass
class ResultEnvelope(Envelope):
    result: Result


############################################################
# Events

@dataclass
class NewBlockEvent(Event, BlockHeader):
    pass


@dataclass
class MatchEvent(Event, MatchEnvelope):
    pass


@dataclass
class ResultEvent(Event, ResultEnvelope):
    pass


@dataclass
class EventEnvelope(Envelope):
    events: List[Event]


############################################################
# RPC

class _RPC:
    together_updateOffer = RPCEndpoint("together_updateOffer")
    together_updateResult = RPCEndpoint("together_updateResult")


class TogetherComputer(Module):
    updateOffer: Method[Callable[[OfferEnvelope, Optional[str]], str]] = Method(
        _RPC.together_updateOffer,
        mungers=[default_root_munger],
    )
    updateResult: Method[Callable[[ResultEnvelope, Optional[str]], str]] = Method(
        _RPC.together_updateResult,
        mungers=[default_root_munger],
    )


class TogetherComputerProtocol(Protocol):
    def updateOffer(self, offer: Dict[str, Any], subscription_id: Optional[str]) -> str:
        pass

    def updateResult(self, offer: Dict[str, Any], subscription_id: Optional[str]) -> str:
        pass
