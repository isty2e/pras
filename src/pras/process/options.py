"""Processing options for the redesigned PRAS package."""

from collections.abc import Iterable
from dataclasses import dataclass

from typing_extensions import Self

from pras.model.enums import (
    AnalysisKind,
    HydrogenPolicy,
    LigandPolicy,
    MutationPolicy,
    OccupancyPolicy,
)
from pras.packing.types import PackingSpec


@dataclass(frozen=True, slots=True)
class ProcessOptions:
    """Normalized processing request."""

    occupancy_policy: OccupancyPolicy = OccupancyPolicy.HIGHEST
    mutation_policy: MutationPolicy = MutationPolicy.HIGHEST_OCCUPANCY
    ligand_policy: LigandPolicy = LigandPolicy.DROP
    hydrogen_policy: HydrogenPolicy = HydrogenPolicy.PRESERVE
    selected_chain_ids: tuple[str, ...] | None = None
    sidechain_packing: PackingSpec | None = None
    protonate_histidines: bool = False
    analyses: frozenset[AnalysisKind] = frozenset()

    def __post_init__(self) -> None:
        selected_chain_ids = self.selected_chain_ids
        if selected_chain_ids is not None:
            normalized_chain_ids: list[str] = []
            seen_chain_ids: set[str] = set()
            for chain_id in selected_chain_ids:
                normalized_chain_id = chain_id.strip()
                if not normalized_chain_id:
                    raise ValueError("selected chain ids must not contain blanks")
                if normalized_chain_id not in seen_chain_ids:
                    normalized_chain_ids.append(normalized_chain_id)
                    seen_chain_ids.add(normalized_chain_id)
            selected_chain_ids = tuple(normalized_chain_ids) or None

        analyses = frozenset(self.analyses)

        object.__setattr__(self, "selected_chain_ids", selected_chain_ids)
        object.__setattr__(self, "analyses", analyses)

    def selects_chain(self, chain_id: str) -> bool:
        """Return whether a chain should be processed under these options."""

        if self.selected_chain_ids is None:
            return True

        return chain_id.strip() in self.selected_chain_ids

    def requests_analysis(self, analysis_kind: AnalysisKind) -> bool:
        """Return whether an analysis kind is enabled."""

        return analysis_kind in self.analyses

    def requests_sidechain_packing(self) -> bool:
        """Return whether side-chain packing was requested."""

        return self.sidechain_packing is not None

    def with_selected_chains(self, chain_ids: Iterable[str] | None) -> Self:
        """Return a copy with updated chain selection."""

        normalized_chain_ids = None if chain_ids is None else tuple(chain_ids)
        return type(self)(
            occupancy_policy=self.occupancy_policy,
            mutation_policy=self.mutation_policy,
            ligand_policy=self.ligand_policy,
            hydrogen_policy=self.hydrogen_policy,
            selected_chain_ids=normalized_chain_ids,
            sidechain_packing=self.sidechain_packing,
            protonate_histidines=self.protonate_histidines,
            analyses=self.analyses,
        )

    def with_sidechain_packing(
        self,
        sidechain_packing: PackingSpec | None,
    ) -> Self:
        """Return a copy with updated side-chain packing configuration."""

        return type(self)(
            occupancy_policy=self.occupancy_policy,
            mutation_policy=self.mutation_policy,
            ligand_policy=self.ligand_policy,
            hydrogen_policy=self.hydrogen_policy,
            selected_chain_ids=self.selected_chain_ids,
            sidechain_packing=sidechain_packing,
            protonate_histidines=self.protonate_histidines,
            analyses=self.analyses,
        )

    def with_requested_analysis(self, analysis_kind: AnalysisKind) -> Self:
        """Return a copy with one additional analysis enabled."""

        return type(self)(
            occupancy_policy=self.occupancy_policy,
            mutation_policy=self.mutation_policy,
            ligand_policy=self.ligand_policy,
            hydrogen_policy=self.hydrogen_policy,
            selected_chain_ids=self.selected_chain_ids,
            sidechain_packing=self.sidechain_packing,
            protonate_histidines=self.protonate_histidines,
            analyses=frozenset((*self.analyses, analysis_kind)),
        )
