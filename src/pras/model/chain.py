"""Chain-level domain entities for the redesigned PRAS package."""

from collections.abc import Iterator
from dataclasses import dataclass

from typing_extensions import Self

from pras.errors import ModelInvariantError, ResidueNotFoundError
from pras.model.atom import Atom
from pras.model.ids import ResidueId
from pras.model.residue import Residue


@dataclass(frozen=True, slots=True)
class Chain:
    """Canonical chain instance."""

    chain_id: str
    residues: tuple[Residue, ...]

    def __post_init__(self) -> None:
        chain_id = self.chain_id.strip()
        residues = tuple(self.residues)

        if not chain_id:
            raise ValueError("chain_id must not be blank")

        residue_ids = tuple(residue.residue_id for residue in residues)
        if len(residue_ids) != len(set(residue_ids)):
            raise ModelInvariantError(
                f"chain {chain_id} contains duplicate residue ids"
            )

        for residue in residues:
            if residue.residue_id.chain_id != chain_id:
                raise ModelInvariantError(
                    f"residue {residue.residue_id.display_token()} does not belong "
                    f"to chain {chain_id}"
                )

        object.__setattr__(self, "chain_id", chain_id)
        object.__setattr__(self, "residues", residues)

    def residue_ids(self) -> tuple[ResidueId, ...]:
        """Return residue identifiers in chain order."""

        return tuple(residue.residue_id for residue in self.residues)

    def has_residue(self, residue_id: ResidueId) -> bool:
        """Return whether a residue exists in the chain."""

        return residue_id in self.residue_ids()

    def residue(self, residue_id: ResidueId) -> Residue:
        """Return a residue by identifier or raise if it is absent."""

        for residue in self.residues:
            if residue.residue_id == residue_id:
                return residue

        raise ResidueNotFoundError(
            f"chain {self.chain_id} has no residue {residue_id.display_token()}"
        )

    def with_updated_residue(self, residue: Residue) -> Self:
        """Return a copy with one residue replaced by identifier."""

        updated_residues = list(self.residues)
        for index, current_residue in enumerate(updated_residues):
            if current_residue.residue_id == residue.residue_id:
                updated_residues[index] = residue
                return type(self)(
                    chain_id=self.chain_id, residues=tuple(updated_residues)
                )

        raise ResidueNotFoundError(
            f"chain {self.chain_id} has no residue {residue.residue_id.display_token()}"
        )

    def residue_window(self, center: ResidueId, radius: int) -> tuple[Residue, ...]:
        """Return a contiguous residue window around a center residue."""

        if radius < 0:
            raise ValueError("radius must be non-negative")

        for index, residue in enumerate(self.residues):
            if residue.residue_id == center:
                start = max(index - radius, 0)
                end = min(index + radius + 1, len(self.residues))
                return self.residues[start:end]

        raise ResidueNotFoundError(
            f"chain {self.chain_id} has no residue {center.display_token()}"
        )

    def iter_atoms(self) -> Iterator[Atom]:
        """Iterate over all atoms in chain order."""

        for residue in self.residues:
            yield from residue.atoms
