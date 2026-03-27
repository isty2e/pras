"""Public API surface for the ``pras`` package."""

from pathlib import Path

from pras.model import ProcessResult, ProteinStructure
from pras.process import ProcessOptions
from pras.workflow.process import process_structure_source


def process_structure(
    source: Path | str | ProteinStructure,
    options: ProcessOptions | None = None,
) -> ProcessResult:
    """Process one structure source through the current workflow spine."""

    return process_structure_source(source, options=options)


__all__ = ["process_structure"]
