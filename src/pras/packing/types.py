"""Canonical request and result entities for side-chain packing backends."""

from dataclasses import dataclass
from enum import Enum

from pras.model.events import ValidationIssue
from pras.model.ids import ResidueId
from pras.model.structure import ProteinStructure


class PackingScope(str, Enum):
    """How much of a structure a packing backend may modify."""

    FULL = "full"
    LOCAL = "local"


class PackingMode(str, Enum):
    """Operational intent for a side-chain packing backend."""

    PACK = "pack"
    REFINE = "refine"


@dataclass(frozen=True, slots=True)
class PackingSpec:
    """Normalized request-level configuration for side-chain packing."""

    backend_name: str
    mode: PackingMode = PackingMode.PACK
    scope: PackingScope = PackingScope.FULL
    target_sequence: str | None = None
    mutable_residue_ids: tuple[ResidueId, ...] | None = None
    frozen_residue_ids: tuple[ResidueId, ...] | None = None

    def __post_init__(self) -> None:
        backend_name = self.backend_name.strip().lower()
        if not backend_name:
            raise ValueError("packing backend_name must not be blank")

        target_sequence = self.target_sequence
        if target_sequence is not None:
            target_sequence = "".join(target_sequence.split()).upper()
            if not target_sequence:
                raise ValueError("packing target_sequence must not be blank")
            if not target_sequence.isalpha():
                raise ValueError(
                    "packing target_sequence must contain only alphabetic codes"
                )

        mutable_residue_ids = _normalize_residue_id_tuple(self.mutable_residue_ids)
        frozen_residue_ids = _normalize_residue_id_tuple(self.frozen_residue_ids)
        mutable_residue_id_set = (
            set(mutable_residue_ids) if mutable_residue_ids is not None else set()
        )
        frozen_residue_id_set = (
            set(frozen_residue_ids) if frozen_residue_ids is not None else set()
        )

        if mutable_residue_id_set & frozen_residue_id_set:
            raise ValueError(
                "packing mutable_residue_ids and frozen_residue_ids must not overlap"
            )

        if self.scope is PackingScope.LOCAL and mutable_residue_ids is None:
            raise ValueError(
                "local side-chain packing requires mutable_residue_ids"
            )

        object.__setattr__(self, "backend_name", backend_name)
        object.__setattr__(self, "target_sequence", target_sequence)
        object.__setattr__(self, "mutable_residue_ids", mutable_residue_ids)
        object.__setattr__(self, "frozen_residue_ids", frozen_residue_ids)

    def has_sequence_override(self) -> bool:
        """Return whether the spec includes a sequence override."""

        return self.target_sequence is not None

    def is_local(self) -> bool:
        """Return whether the request targets a local packing region."""

        return self.scope is PackingScope.LOCAL

    def referenced_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return mutable and frozen residue identifiers in first-seen order."""

        ordered_residue_ids: list[ResidueId] = []
        seen_residue_ids: set[ResidueId] = set()

        for residue_ids in (self.mutable_residue_ids, self.frozen_residue_ids):
            if residue_ids is None:
                continue

            for residue_id in residue_ids:
                if residue_id not in seen_residue_ids:
                    ordered_residue_ids.append(residue_id)
                    seen_residue_ids.add(residue_id)

        return tuple(ordered_residue_ids)

    def references_residue(self, residue_id: ResidueId) -> bool:
        """Return whether a residue is named in the packing specification."""

        return residue_id in self.referenced_residue_ids()


@dataclass(frozen=True, slots=True)
class PackingCapabilities:
    """Declared capabilities of one side-chain packing backend."""

    supports_full_structure_packing: bool
    supports_local_packing: bool
    supports_partial_sequence: bool
    supports_refinement: bool
    supports_noncanonical_components: bool
    deterministic_given_same_inputs: bool

    def supports_spec(self, spec: PackingSpec) -> bool:
        """Return whether these capabilities support one packing spec."""

        if (
            spec.scope is PackingScope.FULL
            and not self.supports_full_structure_packing
        ):
            return False

        if spec.scope is PackingScope.LOCAL and not self.supports_local_packing:
            return False

        if spec.mode is PackingMode.REFINE and not self.supports_refinement:
            return False

        if spec.has_sequence_override() and not self.supports_partial_sequence:
            return False

        return True

    def require_support_for(self, spec: PackingSpec) -> None:
        """Raise when a packing spec exceeds these capabilities."""

        if (
            spec.scope is PackingScope.FULL
            and not self.supports_full_structure_packing
        ):
            raise ValueError(
                "packing backend does not support full-structure packing"
            )

        if spec.scope is PackingScope.LOCAL and not self.supports_local_packing:
            raise ValueError("packing backend does not support local packing")

        if spec.mode is PackingMode.REFINE and not self.supports_refinement:
            raise ValueError("packing backend does not support refinement mode")

        if spec.has_sequence_override() and not self.supports_partial_sequence:
            raise ValueError(
                "packing backend does not support sequence overrides"
            )


@dataclass(frozen=True, slots=True)
class PackingRequest:
    """Normalized internal request for one side-chain packing operation."""

    structure: ProteinStructure
    spec: PackingSpec

    def __post_init__(self) -> None:
        referenced_residue_ids = self.spec.referenced_residue_ids()
        for residue_id in referenced_residue_ids:
            self.structure.residue(residue_id)

        if (
            self.spec.target_sequence is not None
            and self.spec.mutable_residue_ids is not None
            and len(self.spec.target_sequence)
            != len(self.spec.mutable_residue_ids)
        ):
            raise ValueError(
                "packing target_sequence length must match mutable_residue_ids"
            )

    def referenced_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return residue identifiers explicitly referenced by the request."""

        return self.spec.referenced_residue_ids()

    def referenced_residue_count(self) -> int:
        """Return the number of explicitly referenced residues."""

        return len(self.referenced_residue_ids())

    def assert_supported_by(self, capabilities: PackingCapabilities) -> None:
        """Raise when a backend capability set cannot satisfy this request."""

        capabilities.require_support_for(self.spec)


@dataclass(frozen=True, slots=True)
class PackingResult:
    """Structured result from one side-chain packing backend."""

    packed_structure: ProteinStructure
    changed_residue_ids: tuple[ResidueId, ...]
    issues: tuple[ValidationIssue, ...]
    backend_name: str
    backend_version: str | None = None

    def __post_init__(self) -> None:
        backend_name = self.backend_name.strip().lower()
        if not backend_name:
            raise ValueError("packing result backend_name must not be blank")

        backend_version = self.backend_version
        if backend_version is not None:
            backend_version = backend_version.strip() or None

        changed_residue_ids = _normalize_residue_id_tuple(self.changed_residue_ids)
        issues = tuple(self.issues)

        if changed_residue_ids is not None:
            for residue_id in changed_residue_ids:
                self.packed_structure.residue(residue_id)

        object.__setattr__(self, "backend_name", backend_name)
        object.__setattr__(
            self,
            "changed_residue_ids",
            () if changed_residue_ids is None else changed_residue_ids,
        )
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "backend_version", backend_version)

    def changed_residue_count(self) -> int:
        """Return the number of residues changed by the backend."""

        return len(self.changed_residue_ids)

    def changed_residue(self, residue_id: ResidueId) -> bool:
        """Return whether one residue identifier was changed."""

        return residue_id in self.changed_residue_ids

    def has_issues(self) -> bool:
        """Return whether the backend reported any validation issue."""

        return bool(self.issues)


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
