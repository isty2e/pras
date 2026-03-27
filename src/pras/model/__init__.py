"""Canonical domain entities for the redesigned PRAS package."""

from pras.model.atom import Atom, Vec3
from pras.model.chain import Chain
from pras.model.enums import (
    AnalysisKind,
    FileFormat,
    HydrogenPolicy,
    IssueSeverity,
    LigandPolicy,
    MutationPolicy,
    OccupancyPolicy,
    RepairEventKind,
    ValidationIssueKind,
)
from pras.model.events import RepairEvent, ValidationIssue
from pras.model.ids import ResidueId
from pras.model.residue import Residue
from pras.model.result import (
    AnalysisBundle,
    ProcessResult,
    RamachandranAnalysis,
    RamachandranPoint,
    SecondaryStructureAnalysis,
    SecondaryStructureAssignment,
)
from pras.model.structure import ProteinStructure

__all__ = [
    "AnalysisBundle",
    "AnalysisKind",
    "Atom",
    "Chain",
    "FileFormat",
    "HydrogenPolicy",
    "IssueSeverity",
    "LigandPolicy",
    "MutationPolicy",
    "OccupancyPolicy",
    "ProcessResult",
    "ProteinStructure",
    "RamachandranAnalysis",
    "RamachandranPoint",
    "RepairEvent",
    "RepairEventKind",
    "Residue",
    "ResidueId",
    "SecondaryStructureAnalysis",
    "SecondaryStructureAssignment",
    "ValidationIssue",
    "ValidationIssueKind",
    "Vec3",
]
