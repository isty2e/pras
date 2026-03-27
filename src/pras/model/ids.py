"""Identifier value objects for the redesigned PRAS domain model."""

from dataclasses import dataclass

from typing_extensions import Self


@dataclass(frozen=True, order=True, slots=True)
class ResidueId:
    """Canonical identifier for a residue in a chain."""

    chain_id: str
    seq_num: int
    insertion_code: str | None = None

    def __post_init__(self) -> None:
        chain_id = self.chain_id.strip()
        if not chain_id:
            raise ValueError("chain_id must not be blank")

        insertion_code = self.insertion_code
        if insertion_code is not None:
            insertion_code = insertion_code.strip() or None

        object.__setattr__(self, "chain_id", chain_id)
        object.__setattr__(self, "insertion_code", insertion_code)

    def display_token(self) -> str:
        """Return a compact human-readable residue token."""

        insertion = self.insertion_code or ""
        return f"{self.chain_id}:{self.seq_num}{insertion}"

    def with_chain_id(self, chain_id: str) -> Self:
        """Return a copy with a different chain identifier."""

        return type(self)(
            chain_id=chain_id,
            seq_num=self.seq_num,
            insertion_code=self.insertion_code,
        )
