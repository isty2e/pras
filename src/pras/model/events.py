"""Structured events and validation results for the redesigned PRAS package."""

from dataclasses import dataclass

from pras.model.enums import IssueSeverity, RepairEventKind, ValidationIssueKind
from pras.model.ids import ResidueId


@dataclass(frozen=True, slots=True)
class RepairEvent:
    """Structured record of a successful repair or normalization event."""

    kind: RepairEventKind
    residue_id: ResidueId
    component_id: str
    atom_names: tuple[str, ...]
    details: str | None = None

    def __post_init__(self) -> None:
        atom_names = tuple(atom_name.strip().upper() for atom_name in self.atom_names)
        object.__setattr__(self, "atom_names", atom_names)

    def affects_atom(self, atom_name: str) -> bool:
        """Return whether the event references a specific atom."""

        return atom_name.strip().upper() in self.atom_names


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """Structured validation issue attached to a residue or structure."""

    kind: ValidationIssueKind
    severity: IssueSeverity
    message: str
    residue_id: ResidueId | None = None

    def is_error(self) -> bool:
        """Return whether the issue is error-severity."""

        return self.severity is IssueSeverity.ERROR

    def is_warning(self) -> bool:
        """Return whether the issue is warning-severity."""

        return self.severity is IssueSeverity.WARNING
