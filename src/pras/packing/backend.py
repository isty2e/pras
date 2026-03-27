"""Protocol types for side-chain packing backend implementations."""

from typing import Protocol

from pras.packing.types import PackingCapabilities, PackingRequest, PackingResult


class SidechainPackingBackend(Protocol):
    """Internal protocol implemented by side-chain packing backends."""

    def capabilities(self) -> PackingCapabilities:
        """Return declared backend capabilities."""

        ...

    def pack(self, request: PackingRequest) -> PackingResult:
        """Execute one normalized side-chain packing request."""

        ...
