"""Structure-level domain entities for the redesigned PRAS package."""

from collections.abc import Collection, Iterator
from dataclasses import dataclass

from typing_extensions import Self

from pras.errors import ChainNotFoundError, ModelInvariantError, ResidueNotFoundError
from pras.model.atom import Atom
from pras.model.chain import Chain
from pras.model.enums import FileFormat
from pras.model.ids import ResidueId
from pras.model.residue import Residue


@dataclass(frozen=True, slots=True)
class ProteinStructure:
    """Canonical protein structure instance."""

    chains: tuple[Chain, ...]
    ligands: tuple[Residue, ...]
    source_format: FileFormat
    source_name: str | None = None

    def __post_init__(self) -> None:
        chains = tuple(self.chains)
        ligands = tuple(self.ligands)
        chain_ids = tuple(chain.chain_id for chain in chains)

        if len(chain_ids) != len(set(chain_ids)):
            raise ModelInvariantError("structure contains duplicate chain ids")

        if self.source_name is not None:
            source_name = self.source_name.strip() or None
            object.__setattr__(self, "source_name", source_name)

        object.__setattr__(self, "chains", chains)
        object.__setattr__(self, "ligands", ligands)

    def chain_ids(self) -> tuple[str, ...]:
        """Return chain identifiers in structure order."""

        return tuple(chain.chain_id for chain in self.chains)

    def has_chain(self, chain_id: str) -> bool:
        """Return whether the structure contains a chain."""

        normalized_chain_id = chain_id.strip()
        return normalized_chain_id in self.chain_ids()

    def chain(self, chain_id: str) -> Chain:
        """Return a chain by identifier or raise if it is absent."""

        normalized_chain_id = chain_id.strip()
        for chain in self.chains:
            if chain.chain_id == normalized_chain_id:
                return chain

        raise ChainNotFoundError(f"structure has no chain {normalized_chain_id}")

    def residue(self, residue_id: ResidueId) -> Residue:
        """Return a residue by identifier or raise if it is absent."""

        try:
            return self.chain(residue_id.chain_id).residue(residue_id)
        except ChainNotFoundError as error:
            raise ResidueNotFoundError(
                f"structure has no residue {residue_id.display_token()}"
            ) from error

    def select_chains(self, chain_ids: Collection[str]) -> Self:
        """Return a copy containing only selected chains in the given order."""

        normalized_chain_ids = tuple(chain_id.strip() for chain_id in chain_ids)
        selected_chains = tuple(
            self.chain(chain_id) for chain_id in normalized_chain_ids
        )
        return type(self)(
            chains=selected_chains,
            ligands=self.ligands,
            source_format=self.source_format,
            source_name=self.source_name,
        )

    def with_updated_chain(self, chain: Chain) -> Self:
        """Return a copy with one chain replaced by identifier."""

        updated_chains = list(self.chains)
        for index, current_chain in enumerate(updated_chains):
            if current_chain.chain_id == chain.chain_id:
                updated_chains[index] = chain
                return type(self)(
                    chains=tuple(updated_chains),
                    ligands=self.ligands,
                    source_format=self.source_format,
                    source_name=self.source_name,
                )

        raise ChainNotFoundError(f"structure has no chain {chain.chain_id}")

    def iter_residues(self, include_ligands: bool = False) -> Iterator[Residue]:
        """Iterate over residues in chain order."""

        for chain in self.chains:
            yield from chain.residues

        if include_ligands:
            yield from self.ligands

    def iter_atoms(self, include_ligands: bool = False) -> Iterator[Atom]:
        """Iterate over atoms in structure order."""

        for residue in self.iter_residues(include_ligands=include_ligands):
            yield from residue.atoms
