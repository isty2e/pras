"""Rich planning entities for side-chain packing execution."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from typing_extensions import Self

from pras.errors import PackingError
from pras.model.ids import ResidueId
from pras.model.residue import Residue
from pras.model.structure import ProteinStructure
from pras.packing.types import PackingRequest, PackingScope, PackingSpec


@dataclass(frozen=True, slots=True)
class PackingAlphabet:
    """Mapping between component identifiers and one-letter packing tokens."""

    token_by_component_id: Mapping[str, str]

    def __post_init__(self) -> None:
        normalized_mapping: dict[str, str] = {}
        for component_id, token in self.token_by_component_id.items():
            normalized_component_id = component_id.strip().upper()
            normalized_token = token.strip().upper()
            if not normalized_component_id:
                raise ValueError("packing alphabet component ids must not be blank")

            if len(normalized_token) != 1 or not normalized_token.isalpha():
                raise ValueError(
                    "packing alphabet tokens must be one alphabetic character"
                )

            normalized_mapping[normalized_component_id] = normalized_token

        object.__setattr__(
            self,
            "token_by_component_id",
            MappingProxyType(normalized_mapping),
        )

    def supports_component(self, component_id: str) -> bool:
        """Return whether the alphabet supports a component identifier."""

        return component_id.strip().upper() in self.token_by_component_id

    def require_token(self, component_id: str) -> str:
        """Return the token for one component identifier or raise."""

        normalized_component_id = component_id.strip().upper()
        token = self.token_by_component_id.get(normalized_component_id)
        if token is None:
            raise PackingError(
                f"packing alphabet does not support component {normalized_component_id}"
            )

        return token

    def sequence_for_residues(
        self, residues: Iterable[Residue]
    ) -> tuple[str, ...]:
        """Return one tokenized sequence for residues in order."""

        return tuple(self.require_token(residue.component_id) for residue in residues)


@dataclass(frozen=True, slots=True)
class PackingSelection:
    """Resolved mutable and fixed residue selection for one packing request."""

    scope: PackingScope
    polymer_residue_ids: tuple[ResidueId, ...]
    mutable_residue_ids: tuple[ResidueId, ...] | None = None
    frozen_residue_ids: tuple[ResidueId, ...] | None = None

    def __post_init__(self) -> None:
        polymer_residue_ids = _normalize_residue_id_tuple(self.polymer_residue_ids)
        if polymer_residue_ids is None:
            raise ValueError("packing selection requires at least one polymer residue")

        mutable_residue_ids = _normalize_residue_id_tuple(self.mutable_residue_ids)
        frozen_residue_ids = _normalize_residue_id_tuple(self.frozen_residue_ids)
        polymer_residue_id_set = set(polymer_residue_ids)

        for residue_ids, label in (
            (mutable_residue_ids, "mutable_residue_ids"),
            (frozen_residue_ids, "frozen_residue_ids"),
        ):
            if residue_ids is None:
                continue

            unknown_ids = tuple(
                residue_id
                for residue_id in residue_ids
                if residue_id not in polymer_residue_id_set
            )
            if unknown_ids:
                raise ValueError(
                    f"packing selection {label} must belong to polymer residues"
                )

        if mutable_residue_ids is not None and frozen_residue_ids is not None:
            if set(mutable_residue_ids) & set(frozen_residue_ids):
                raise ValueError(
                    "packing selection mutable and frozen residues must not overlap"
                )

        object.__setattr__(self, "polymer_residue_ids", polymer_residue_ids)
        object.__setattr__(self, "mutable_residue_ids", mutable_residue_ids)
        object.__setattr__(self, "frozen_residue_ids", frozen_residue_ids)

    def selected_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return residue identifiers explicitly targeted by the request."""

        if self.mutable_residue_ids is not None:
            return self.mutable_residue_ids

        return self.polymer_residue_ids

    def fixed_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return residue identifiers that must remain fixed during packing."""

        ordered_residue_ids: list[ResidueId] = []
        seen_residue_ids: set[ResidueId] = set()
        selected_residue_id_set = set(self.selected_residue_ids())

        if self.scope is PackingScope.LOCAL:
            for residue_id in self.polymer_residue_ids:
                if residue_id not in selected_residue_id_set:
                    ordered_residue_ids.append(residue_id)
                    seen_residue_ids.add(residue_id)

        if self.frozen_residue_ids is not None:
            for residue_id in self.frozen_residue_ids:
                if residue_id not in seen_residue_ids:
                    ordered_residue_ids.append(residue_id)
                    seen_residue_ids.add(residue_id)

        return tuple(ordered_residue_ids)

    def is_selected(self, residue_id: ResidueId) -> bool:
        """Return whether one residue is explicitly targeted."""

        return residue_id in self.selected_residue_ids()

    def is_fixed(self, residue_id: ResidueId) -> bool:
        """Return whether one residue must remain fixed."""

        return residue_id in self.fixed_residue_ids()

    def selected_residue_count(self) -> int:
        """Return the number of explicitly targeted residues."""

        return len(self.selected_residue_ids())

    def requires_sequence_override(self) -> bool:
        """Return whether one backend needs an explicit sequence override."""

        return bool(self.fixed_residue_ids())


@dataclass(frozen=True, slots=True)
class PackingPlan:
    """Resolved execution plan for one side-chain packing request."""

    structure: ProteinStructure
    spec: PackingSpec
    selection: PackingSelection

    @classmethod
    def from_request(cls, request: PackingRequest) -> Self:
        """Build one execution plan from a normalized packing request."""

        polymer_residue_ids = tuple(
            residue.residue_id
            for chain in request.structure.chains
            for residue in chain.residues
        )
        selection = PackingSelection(
            scope=request.spec.scope,
            polymer_residue_ids=polymer_residue_ids,
            mutable_residue_ids=request.spec.mutable_residue_ids,
            frozen_residue_ids=request.spec.frozen_residue_ids,
        )
        return cls(
            structure=request.structure,
            spec=request.spec,
            selection=selection,
        )

    def polymer_residues(self) -> tuple[Residue, ...]:
        """Return polymer residues in structure order."""

        return tuple(
            residue for chain in self.structure.chains for residue in chain.residues
        )

    def polymer_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return polymer residue identifiers in structure order."""

        return self.selection.polymer_residue_ids

    def selected_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return residue identifiers explicitly targeted by this plan."""

        return self.selection.selected_residue_ids()

    def fixed_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return residue identifiers that must remain fixed."""

        return self.selection.fixed_residue_ids()

    def selected_residue_count(self) -> int:
        """Return the number of explicitly targeted residues."""

        return self.selection.selected_residue_count()

    def residue(self, residue_id: ResidueId) -> Residue:
        """Return one polymer residue from the plan structure."""

        return self.structure.residue(residue_id)

    def selected_residues(self) -> tuple[Residue, ...]:
        """Return explicitly targeted residues in selection order."""

        return tuple(
            self.residue(residue_id)
            for residue_id in self.selected_residue_ids()
        )

    def original_sequence_tokens(
        self,
        alphabet: PackingAlphabet,
    ) -> tuple[str, ...]:
        """Return the original tokenized polymer sequence."""

        return alphabet.sequence_for_residues(self.polymer_residues())

    def effective_sequence_tokens(
        self,
        alphabet: PackingAlphabet,
    ) -> tuple[str, ...]:
        """Return the tokenized sequence after applying requested mutations."""

        original_tokens = list(self.original_sequence_tokens(alphabet))
        if self.spec.target_sequence is None:
            return tuple(original_tokens)

        replacement_tokens = tuple(self.spec.target_sequence)
        if self.spec.mutable_residue_ids is None:
            if len(replacement_tokens) != len(original_tokens):
                raise PackingError(
                    "full-structure target_sequence must match polymer residue count"
                )
            return replacement_tokens

        replacement_by_residue_id = dict(
            zip(
                self.spec.mutable_residue_ids,
                replacement_tokens,
                strict=True,
            )
        )
        for index, residue_id in enumerate(self.polymer_residue_ids()):
            replacement_token = replacement_by_residue_id.get(residue_id)
            if replacement_token is not None:
                original_tokens[index] = replacement_token

        return tuple(original_tokens)

    def changed_residue_ids_after(
        self,
        packed_structure: ProteinStructure,
    ) -> tuple[ResidueId, ...]:
        """Return residue identifiers whose packed residues differ from the input."""

        packed_residues = tuple(
            residue for chain in packed_structure.chains for residue in chain.residues
        )
        original_residues = self.polymer_residues()
        if len(packed_residues) != len(original_residues):
            raise PackingError(
                "packed structure changed the number of polymer residues unexpectedly"
            )

        changed_residue_ids: list[ResidueId] = []
        for original_residue, packed_residue in zip(
            original_residues,
            packed_residues,
            strict=True,
        ):
            if original_residue.residue_id != packed_residue.residue_id:
                raise PackingError(
                    "packed structure changed residue identifiers or order unexpectedly"
                )

            if original_residue != packed_residue:
                changed_residue_ids.append(packed_residue.residue_id)

        return tuple(changed_residue_ids)


def _normalize_residue_id_tuple(
    residue_ids: tuple[ResidueId, ...] | None,
) -> tuple[ResidueId, ...] | None:
    """Normalize one optional residue-identifier tuple."""

    if residue_ids is None:
        return None

    normalized_residue_ids: list[ResidueId] = []
    seen_residue_ids: set[ResidueId] = set()
    for residue_id in residue_ids:
        if residue_id not in seen_residue_ids:
            normalized_residue_ids.append(residue_id)
            seen_residue_ids.add(residue_id)

    return tuple(normalized_residue_ids) or None
