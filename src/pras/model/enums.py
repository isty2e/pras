"""Enum types for the redesigned PRAS domain model."""

from enum import Enum


class FileFormat(str, Enum):
    """Canonical structure file formats."""

    PDB = "pdb"
    MMCIF = "mmcif"


class OccupancyPolicy(str, Enum):
    """Policy for resolving alternate atom occupancies."""

    HIGHEST = "highest"
    LOWEST = "lowest"


class MutationPolicy(str, Enum):
    """Policy for resolving residue-level mutation conflicts."""

    HIGHEST_OCCUPANCY = "highest_occupancy"
    LOWEST_OCCUPANCY = "lowest_occupancy"


class LigandPolicy(str, Enum):
    """Policy for retaining or dropping ligands."""

    DROP = "drop"
    KEEP = "keep"


class HydrogenPolicy(str, Enum):
    """Policy for hydrogen handling during processing."""

    PRESERVE = "preserve"
    ADD_MISSING = "add_missing"


class AnalysisKind(str, Enum):
    """Analysis modes supported by the redesigned package."""

    SECONDARY_STRUCTURE = "secondary_structure"
    RAMACHANDRAN = "ramachandran"


class RepairEventKind(str, Enum):
    """Kinds of repairs and canonicalization events."""

    HEAVY_ATOMS_ADDED = "heavy_atoms_added"
    HYDROGENS_ADDED = "hydrogens_added"
    C_TERMINAL_OXT_ADDED = "c_terminal_oxt_added"
    COMPONENT_NORMALIZED = "component_normalized"


class ValidationIssueKind(str, Enum):
    """Kinds of structural validation issues."""

    MISSING_EXPECTED_ATOMS = "missing_expected_atoms"
    UNEXPECTED_ATOMS = "unexpected_atoms"
    INVALID_BACKBONE = "invalid_backbone"
    UNSUPPORTED_COMPONENT = "unsupported_component"


class IssueSeverity(str, Enum):
    """Validation issue severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
