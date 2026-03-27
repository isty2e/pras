"""Hydrogen placement over the canonical structure model."""

from dataclasses import replace

from pras.chemistry import (
    ComponentLibrary,
    ResidueTemplate,
    build_standard_component_library,
)
from pras.model import (
    Chain,
    IssueSeverity,
    ProcessResult,
    ProteinStructure,
    RepairEvent,
    RepairEventKind,
    Residue,
    ResidueId,
    ValidationIssue,
    ValidationIssueKind,
)
from pras.repair.heavy_atoms import repair_heavy_atoms, residue_from_payload
from pras.repair.hydrogen_engine import (
    generate_hydrogen_payload,
    histidine_delta_hydrogen,
    n_terminal_hydrogen_coordinates,
)
from pras.repair.payloads import (
    BackboneHydrogenPlacement,
    HydrogenationContext,
    OrderedAtomPayload,
    ResiduePayload,
    RotatableHydrogenEnvironment,
)


def add_hydrogens(
    structure: ProteinStructure,
    component_library: ComponentLibrary | None = None,
    reference_structure: ProteinStructure | None = None,
    *,
    protonate_histidines: bool = False,
) -> ProcessResult:
    """Add hydrogens to supported chains while preserving input ligands."""

    library = (
        build_standard_component_library()
        if component_library is None
        else component_library
    )
    heavy_result = repair_heavy_atoms(
        structure,
        component_library=library,
        reference_structure=reference_structure,
    )
    heavy_structure = ProteinStructure(
        chains=heavy_result.structure.chains,
        ligands=structure.ligands,
        source_format=structure.source_format,
        source_name=structure.source_name,
    )
    repaired_chains: list[Chain] = []
    repairs = list(heavy_result.repairs)
    issues = list(heavy_result.issues)

    for chain in heavy_structure.chains:
        unsupported_residues = unsupported_hydrogenation_residues(chain, library)
        if unsupported_residues:
            issues.extend(
                hydrogenation_unsupported_issues(
                    unsupported_residues,
                    chain_id=chain.chain_id,
                )
            )
            repaired_chains.append(chain)
            continue

        hydrogenated_chain, chain_repairs = hydrogenate_chain(
            chain,
            protonate_histidines=protonate_histidines,
            component_library=library,
        )
        repaired_chains.append(hydrogenated_chain)
        repairs.extend(chain_repairs)

    repaired_structure = ProteinStructure(
        chains=tuple(repaired_chains),
        ligands=structure.ligands,
        source_format=structure.source_format,
        source_name=structure.source_name,
    )
    return ProcessResult(
        structure=repaired_structure,
        repairs=tuple(repairs),
        issues=tuple(issues),
        analyses=None,
    )


def unsupported_hydrogenation_residues(
    chain: Chain, component_library: ComponentLibrary
) -> tuple[Residue, ...]:
    """Return residues in a chain that cannot be hydrogenated."""

    unsupported: list[Residue] = []
    for residue in chain.residues:
        template = component_library.get(residue.component_id)
        if template is None or not template.can_add_hydrogens():
            unsupported.append(residue)

    return tuple(unsupported)


def hydrogenation_unsupported_issues(
    residues: tuple[Residue, ...], *, chain_id: str
) -> tuple[ValidationIssue, ...]:
    """Return warnings for a chain skipped during hydrogen placement."""

    return tuple(
        ValidationIssue(
            kind=ValidationIssueKind.UNSUPPORTED_COMPONENT,
            severity=IssueSeverity.WARNING,
            residue_id=residue.residue_id,
            message=(
                f"chain {chain_id} contains unsupported component "
                f"{residue.component_id}; leaving chain unhydrogenated"
            ),
        )
        for residue in residues
    )


def hydrogenate_chain(
    chain: Chain,
    *,
    protonate_histidines: bool,
    component_library: ComponentLibrary,
) -> tuple[Chain, tuple[RepairEvent, ...]]:
    """Hydrogenate one supported chain and return chain-local repair events."""

    if not chain.residues:
        return chain, ()

    payloads = [residue_payload(residue) for residue in chain.residues]
    templates = [
        component_library.require(residue.component_id)
        for residue in chain.residues
    ]
    residue_positions: list[list[list[float]]] = []
    residue_atom_names: list[list[str]] = []
    backbone_hydrogens: list[BackboneHydrogenPlacement] = []

    atom_names_by_residue = [list(residue.atom_names()) for residue in chain.residues]
    atom_positions_by_residue = [
        residue_coordinates(residue) for residue in chain.residues
    ]
    optimization_residue_numbers = [
        optimization_residue_number(residue) for residue in chain.residues
    ]
    forcefield_parameters_by_residue = tuple(
        template.forcefield_parameters for template in templates
    )
    sg_coordinates = collect_sg_coordinates(chain.residues)
    rotatable_hydrogen_environments = build_rotatable_hydrogen_environments(
        atom_names_by_residue=atom_names_by_residue,
        atom_positions_by_residue=atom_positions_by_residue,
        residue_numbers=optimization_residue_numbers,
        forcefield_parameters_by_residue=forcefield_parameters_by_residue,
    )

    for residue_index, residue in enumerate(chain.residues):
        template = templates[residue_index]
        next_payload = (
            payloads[residue_index + 1]
            if residue_index < len(chain.residues) - 1
            else None
        )
        include_backbone_hydrogen = (
            next_payload is not None
            and chain.residues[residue_index + 1].component_id != "PRO"
        )
        coordinates, atom_names, backbone_hydrogen = hydrogenate_residue(
            residue=residue,
            payload=payloads[residue_index],
            context=HydrogenationContext(
                residue_index=residue_index,
                residue_number=optimization_residue_numbers[residue_index],
                rotatable_hydrogen_environments=rotatable_hydrogen_environments,
                sg_coordinates=sg_coordinates,
                next_payload=next_payload,
                include_backbone_hydrogen=include_backbone_hydrogen,
            ),
            template=template,
        )
        residue_positions.append(coordinates)
        residue_atom_names.append(atom_names)
        if backbone_hydrogen is not None:
            backbone_hydrogens.append(backbone_hydrogen)

    protonated_histidines = (
        protonate_histidine_residues(
            chain.residues,
            payloads,
            residue_positions,
            residue_atom_names,
        )
        if protonate_histidines
        else frozenset()
    )
    apply_backbone_hydrogens(residue_positions, residue_atom_names, backbone_hydrogens)
    apply_n_terminal_hydrogens(
        chain.residues[0],
        payloads[0],
        residue_positions[0],
        residue_atom_names[0],
    )

    hydrogenated_residues: list[Residue] = []
    repairs: list[RepairEvent] = []
    for residue, atom_names, coordinates in zip(
        chain.residues,
        residue_atom_names,
        residue_positions,
        strict=True,
    ):
        hydrogenated_residue = residue_from_payload(
            residue,
            OrderedAtomPayload(
                atom_names=atom_names,
                atom_coordinates=coordinates,
            ),
        )
        hydrogenated_residues.append(hydrogenated_residue)
        added_atoms = tuple(
            atom_name
            for atom_name in hydrogenated_residue.atom_names()
            if atom_name not in residue.atom_names()
        )
        if added_atoms:
            details = None
            if residue.residue_id in protonated_histidines:
                details = "histidine protonation (+1 charge) applied"
            repairs.append(
                RepairEvent(
                    kind=RepairEventKind.HYDROGENS_ADDED,
                    residue_id=residue.residue_id,
                    component_id=residue.component_id,
                    atom_names=added_atoms,
                    details=details,
                )
            )

    return replace(chain, residues=tuple(hydrogenated_residues)), tuple(repairs)


def hydrogenate_residue(
    *,
    residue: Residue,
    payload: ResiduePayload,
    context: HydrogenationContext,
    template: ResidueTemplate,
) -> tuple[list[list[float]], list[str], BackboneHydrogenPlacement | None]:
    """Hydrogenate one residue with the internal hydrogen engine."""

    hydrogen_semantics = template.hydrogen_semantics
    if hydrogen_semantics is None:
        raise ValueError("hydrogen placement requires hydrogen semantics")

    result = generate_hydrogen_payload(
        payload=payload,
        context=context,
        semantics=hydrogen_semantics,
    )

    return (
        result.atom_coordinates,
        result.atom_names,
        result.backbone_hydrogen,
    )


def protonate_histidine_residues(
    residues: tuple[Residue, ...],
    payloads: list[ResiduePayload],
    residue_positions: list[list[list[float]]],
    residue_atom_names: list[list[str]],
) -> frozenset[ResidueId]:
    """Apply the legacy 20%-of-HIS protonation rule without file side effects."""

    histidine_indices = [
        index for index, residue in enumerate(residues) if residue.component_id == "HIS"
    ]
    if len(histidine_indices) <= 4:
        return frozenset()

    protonated_indices = histidine_indices[: len(histidine_indices) // 5]
    protonated_residues: set[ResidueId] = set()
    for index in protonated_indices:
        residue_positions[index].append(histidine_delta_hydrogen(payloads[index]))
        residue_atom_names[index].append("HD1")
        protonated_residues.add(residues[index].residue_id)

    return frozenset(protonated_residues)


def apply_backbone_hydrogens(
    residue_positions: list[list[list[float]]],
    residue_atom_names: list[list[str]],
    backbone_hydrogens: list[BackboneHydrogenPlacement],
) -> None:
    """Append backbone hydrogens to the next residue in chain order."""

    for placement in backbone_hydrogens:
        residue_positions[placement.residue_index + 1].append(placement.coordinates)
        residue_atom_names[placement.residue_index + 1].append("H")


def apply_n_terminal_hydrogens(
    first_residue: Residue,
    payload: ResiduePayload,
    residue_positions: list[list[float]],
    residue_atom_names: list[str],
) -> None:
    """Append N-terminal hydrogens using the matching legacy rule."""

    terminal_hydrogens = n_terminal_hydrogen_coordinates(
        payload, first_residue.component_id
    )
    residue_positions.extend(terminal_hydrogens)
    if first_residue.component_id == "PRO":
        residue_atom_names.extend(["H1", "H2"])
    else:
        residue_atom_names.extend(["H1", "H2", "H3"])


def collect_sg_coordinates(residues: tuple[Residue, ...]) -> list[list[float]]:
    """Collect all SG coordinates in a chain for disulfide-bond checks."""

    return [
        [atom.position.x, atom.position.y, atom.position.z]
        for residue in residues
        if residue.has_atom("SG")
        for atom in (residue.atom("SG"),)
    ]


def build_rotatable_hydrogen_environments(
    *,
    atom_names_by_residue: list[list[str]],
    atom_positions_by_residue: list[list[list[float]]],
    residue_numbers: list[str],
    forcefield_parameters_by_residue,
) -> tuple[RotatableHydrogenEnvironment, ...]:
    """Pack per-residue interaction environments for rotatable hydrogen search."""

    environments: list[RotatableHydrogenEnvironment] = []
    for residue_index, residue_number in enumerate(residue_numbers):
        atom_x: list[float] = []
        atom_y: list[float] = []
        atom_z: list[float] = []
        charges: list[float] = []
        sigmas_nm: list[float] = []
        epsilons_kj_mol: list[float] = []

        for other_index, (
            residue_atom_names,
            residue_positions,
            parameters_by_atom,
        ) in enumerate(
            zip(
                atom_names_by_residue,
                atom_positions_by_residue,
                forcefield_parameters_by_residue,
                strict=True,
            )
        ):
            if other_index == residue_index:
                continue

            for atom_name, atom_position in zip(
                residue_atom_names,
                residue_positions,
                strict=True,
            ):
                atom_parameters = parameters_by_atom.get(atom_name)
                if atom_parameters is None:
                    continue

                atom_x.append(float(atom_position[0]))
                atom_y.append(float(atom_position[1]))
                atom_z.append(float(atom_position[2]))
                charges.append(atom_parameters.charge)
                sigmas_nm.append(atom_parameters.sigma_nm)
                epsilons_kj_mol.append(atom_parameters.epsilon_kj_mol)

        environments.append(
            RotatableHydrogenEnvironment(
                residue_number=residue_number,
                atom_x=tuple(atom_x),
                atom_y=tuple(atom_y),
                atom_z=tuple(atom_z),
                charges=tuple(charges),
                sigmas_nm=tuple(sigmas_nm),
                epsilons_kj_mol=tuple(epsilons_kj_mol),
            )
        )

    return tuple(environments)


def residue_payload(residue: Residue) -> ResiduePayload:
    """Project a canonical residue into the hydrogen payload."""

    return ResiduePayload(
        residue_label=f"{residue.component_id}{residue_sequence_token(residue)}",
        atom_names=list(residue.atom_names()),
        atom_coordinates=residue_coordinates(residue),
    )


def residue_sequence_token(residue: Residue) -> str:
    """Return the residue sequence token including insertion code when present."""

    insertion_code = residue.residue_id.insertion_code or ""
    return f"{residue.residue_id.seq_num}{insertion_code}"


def optimization_residue_number(residue: Residue) -> str:
    """Return the legacy PRAS residue-number token for rotatable-H refinement."""

    return str(residue.residue_id.seq_num)


def residue_coordinates(residue: Residue) -> list[list[float]]:
    """Return residue coordinates as mutable three-float vectors."""

    return [
        [atom.position.x, atom.position.y, atom.position.z] for atom in residue.atoms
    ]
