from typing import Any, Callable, Dict, List, Optional
from typing_extensions import Protocol

from dataclasses import dataclass

from web3.method import Method, default_root_munger
from web3.module import Module
from web3.types import RPCEndpoint

from .computer import Envelope, Instance, ResultEnvelope


@dataclass
class Join:
    group_name: str
    worker_name: str
    host_name: str
    host_ip: str
    interface_ip: List[str]
    instance: Instance
    config: Optional[Dict[str, Any]]


@dataclass
class JoinEnvelope(Envelope):
    join: Join


class _RPC:
    coordinator_join = RPCEndpoint("coordinator_join")
    coordinator_updateResult = RPCEndpoint("coordinator_updateResult")


class TogetherCoordinator(Module):
    join: Method[Callable[[JoinEnvelope, Optional[str]], Dict[str, Any]]] = Method(
        _RPC.coordinator_join,
        mungers=[default_root_munger],
    )
    updateResult: Method[Callable[[ResultEnvelope, Optional[str]], str]] = Method(
        _RPC.coordinator_updateResult,
        mungers=[default_root_munger],
    )


class TogetherCoordinatorProtocol(Protocol):
    def join(self, offer: Dict[str, Any], subscription_id: Optional[str]) -> Dict[str, Any]:
        pass

    def updateResult(self, offer: Dict[str, Any], subscription_id: Optional[str]) -> str:
        pass
