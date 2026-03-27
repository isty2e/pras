"""Internal processing workflow over canonical structures."""

from dataclasses import replace
from pathlib import Path

from pras.io import read_structure
from pras.model import (
    HydrogenPolicy,
    LigandPolicy,
    ProcessResult,
    ProteinStructure,
    ValidationIssue,
)
from pras.packing import PackingRequest
from pras.packing.backend import SidechainPackingBackend
from pras.packing.faspr_backend import FasprPackingBackend
from pras.process import ProcessOptions
from pras.repair import add_hydrogens, repair_heavy_atoms


def process_canonical_structure(
    structure: ProteinStructure,
    options: ProcessOptions | None = None,
) -> ProcessResult:
    """Process a canonical structure through the current repair workflow."""

    normalized_options = ProcessOptions() if options is None else options
    validate_supported_workflow_options(normalized_options)
    normalized_structure = apply_canonical_boundary_options(
        structure,
        options=normalized_options,
    )
    packed_reference, packing_issues = packed_reference_for_workflow(
        normalized_structure,
        normalized_options,
    )

    if should_hydrogenate(normalized_structure, normalized_options):
        result = add_hydrogens(
            normalized_structure,
            reference_structure=packed_reference,
            protonate_histidines=normalized_options.protonate_histidines,
        )
    else:
        heavy_result = repair_heavy_atoms(
            normalized_structure,
            reference_structure=packed_reference,
        )
        if heavy_result.structure.ligands == normalized_structure.ligands:
            result = heavy_result
        else:
            result = heavy_result.with_structure(
                replace(
                    heavy_result.structure,
                    ligands=normalized_structure.ligands,
                )
            )

    if not packing_issues:
        return result

    return replace(
        result,
        issues=result.issues + packing_issues,
    )


def packed_reference_for_workflow(
    structure: ProteinStructure,
    options: ProcessOptions,
) -> tuple[ProteinStructure | None, tuple[ValidationIssue, ...]]:
    """Return an optional packed reference structure for repair guidance."""

    if options.sidechain_packing is None:
        return None, ()

    packing_result = resolve_sidechain_packing_backend(
        options.sidechain_packing.backend_name
    ).pack(
        PackingRequest(
            structure=structure,
            spec=options.sidechain_packing,
        )
    )
    return packing_result.packed_structure, packing_result.issues


def resolve_sidechain_packing_backend(
    backend_name: str,
) -> SidechainPackingBackend:
    """Return the internal backend implementation for one backend name."""

    if backend_name == "faspr":
        return FasprPackingBackend()

    raise NotImplementedError(
        f"side-chain packing backend {backend_name!r} is not implemented"
    )


def process_structure_source(
    source: Path | str | ProteinStructure,
    options: ProcessOptions | None = None,
) -> ProcessResult:
    """Normalize one supported source and process it through the workflow."""

    normalized_options = ProcessOptions() if options is None else options
    structure = normalize_source_structure(source, normalized_options)
    return process_canonical_structure(structure, options=normalized_options)


def normalize_source_structure(
    source: Path | str | ProteinStructure,
    options: ProcessOptions,
) -> ProteinStructure:
    """Normalize one supported source into the canonical structure model."""

    if isinstance(source, ProteinStructure):
        return apply_canonical_boundary_options(source, options=options)

    path = Path(source)
    return read_structure(path, options=options)


def apply_canonical_boundary_options(
    structure: ProteinStructure,
    *,
    options: ProcessOptions,
) -> ProteinStructure:
    """Apply canonical-only boundary options to an already normalized structure."""

    selected_structure = (
        structure.select_chains(options.selected_chain_ids)
        if options.selected_chain_ids is not None
        else structure
    )
    if options.ligand_policy is LigandPolicy.DROP:
        ligands = ()
    elif options.selected_chain_ids is None:
        ligands = selected_structure.ligands
    else:
        selected_chain_ids = frozenset(options.selected_chain_ids)
        ligands = tuple(
            ligand
            for ligand in selected_structure.ligands
            if ligand.residue_id.chain_id in selected_chain_ids
        )

    if selected_structure is structure and ligands == structure.ligands:
        return structure

    return ProteinStructure(
        chains=selected_structure.chains,
        ligands=ligands,
        source_format=selected_structure.source_format,
        source_name=selected_structure.source_name,
    )


def validate_supported_workflow_options(options: ProcessOptions) -> None:
    """Reject option combinations not yet supported by the workflow spine."""

    if options.analyses:
        raise NotImplementedError("Analysis execution is not implemented yet")


def should_hydrogenate(
    structure: ProteinStructure, options: ProcessOptions
) -> bool:
    """Return whether the workflow should run hydrogen placement."""

    return options.hydrogen_policy is HydrogenPolicy.ADD_MISSING
