"""Heavy-atom repair over the canonical structure model."""

from collections.abc import Mapping
from dataclasses import replace

from pras.chemistry import (
    ComponentLibrary,
    ResidueTemplate,
    build_standard_component_library,
)
from pras.model import (
    Atom,
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
    Vec3,
)
from pras.repair.heavy_engine import (
    c_terminal_oxt,
    repair_residue_payload,
)
from pras.repair.payloads import HeavyRepairContext, OrderedAtomPayload

BACKBONE_ATOM_NAMES: tuple[str, ...] = ("N", "CA", "C")
ILE_ATOM_ALIASES: dict[str, str] = {"CD": "CD1"}
TERMINAL_EXCLUDED_ATOMS: frozenset[str] = frozenset({"OXT"})


def repair_heavy_atoms(
    structure: ProteinStructure,
    component_library: ComponentLibrary | None = None,
    reference_structure: ProteinStructure | None = None,
) -> ProcessResult:
    """Repair missing heavy atoms for supported protein residues."""

    library = (
        build_standard_component_library()
        if component_library is None
        else component_library
    )
    stripped_structure = strip_hydrogens(structure)
    stripped_reference_structure = (
        None if reference_structure is None else strip_hydrogens(reference_structure)
    )
    repaired_chains = []
    repairs: list[RepairEvent] = []
    issues: list[ValidationIssue] = []

    for chain in stripped_structure.chains:
        normalized_residues = [
            normalize_residue_inputs(residue, library, repairs)
            for residue in chain.residues
        ]
        reference_residue_by_id = normalized_reference_residue_by_id(
            stripped_reference_structure,
            chain_id=chain.chain_id,
            component_library=library,
        )
        repaired_residues: list[Residue] = []

        for residue_index, residue in enumerate(normalized_residues):
            template = library.get(residue.component_id)
            if template is None:
                issues.append(
                    ValidationIssue(
                        kind=ValidationIssueKind.UNSUPPORTED_COMPONENT,
                        severity=IssueSeverity.WARNING,
                        residue_id=residue.residue_id,
                        message=(
                            f"{residue.residue_id.display_token()} has unsupported "
                            f"component {residue.component_id}; leaving residue "
                            "unchanged"
                        ),
                    )
                )
                repaired_residues.append(residue)
                continue

            missing_atoms = missing_atoms_for_repair(residue, template)
            if missing_atoms and any(
                atom_name in BACKBONE_ATOM_NAMES for atom_name in missing_atoms
            ):
                issues.append(
                    ValidationIssue(
                        kind=ValidationIssueKind.INVALID_BACKBONE,
                        severity=IssueSeverity.ERROR,
                        residue_id=residue.residue_id,
                        message=(
                            f"{residue.residue_id.display_token()} is missing required "
                            "backbone atoms and cannot be repaired"
                        ),
                    )
                )
                repaired_residues.append(residue)
                continue

            if missing_atoms and not template.can_repair_heavy_atoms():
                issues.append(
                    ValidationIssue(
                        kind=ValidationIssueKind.UNSUPPORTED_COMPONENT,
                        severity=IssueSeverity.WARNING,
                        residue_id=residue.residue_id,
                        message=(
                            f"{residue.residue_id.display_token()} requires heavy-atom "
                            f"repair for {residue.component_id}, but that builder "
                            "is not "
                            "implemented yet; leaving residue unchanged"
                        ),
                    )
                )
                repaired_residues.append(residue)
                continue

            previous_residue = (
                repaired_residues[-1] if repaired_residues else normalized_residues[-1]
            )
            next_residue = normalized_residues[
                (residue_index + 1) % len(normalized_residues)
            ]
            repaired_residue, added_atoms = repair_residue(
                residue=residue,
                previous_residue=previous_residue,
                next_residue=next_residue,
                missing_atoms=missing_atoms,
                template=template,
                reference_residue=reference_residue_by_id.get(residue.residue_id),
            )
            if added_atoms:
                repairs.append(
                    RepairEvent(
                        kind=RepairEventKind.HEAVY_ATOMS_ADDED,
                        residue_id=repaired_residue.residue_id,
                        component_id=repaired_residue.component_id,
                        atom_names=tuple(added_atoms),
                    )
                )

            repaired_residues.append(repaired_residue)

        if repaired_residues and not repaired_residues[-1].has_atom("OXT"):
            repaired_terminal_residue = add_c_terminal_oxt(repaired_residues[-1])
            repairs.append(
                RepairEvent(
                    kind=RepairEventKind.C_TERMINAL_OXT_ADDED,
                    residue_id=repaired_terminal_residue.residue_id,
                    component_id=repaired_terminal_residue.component_id,
                    atom_names=("OXT",),
                )
            )
            repaired_residues[-1] = repaired_terminal_residue

        repaired_chains.append(replace(chain, residues=tuple(repaired_residues)))

    repaired_structure = ProteinStructure(
        chains=tuple(repaired_chains),
        ligands=stripped_structure.ligands,
        source_format=stripped_structure.source_format,
        source_name=stripped_structure.source_name,
    )
    return ProcessResult(
        structure=repaired_structure,
        repairs=tuple(repairs),
        issues=tuple(issues),
        analyses=None,
    )


def strip_hydrogens(structure: ProteinStructure) -> ProteinStructure:
    """Return a copy of the structure without hydrogen atoms."""

    stripped_chains: list[Chain] = []
    structure_changed = False
    for chain in structure.chains:
        stripped_residues = []
        chain_changed = False
        for residue in chain.residues:
            stripped_residue = strip_hydrogens_from_residue(residue)
            if stripped_residue is not residue:
                chain_changed = True
            stripped_residues.append(stripped_residue)

        if chain_changed:
            structure_changed = True
            stripped_chains.append(replace(chain, residues=tuple(stripped_residues)))
            continue

        stripped_chains.append(chain)

    if not structure_changed:
        return structure

    return ProteinStructure(
        chains=tuple(stripped_chains),
        ligands=structure.ligands,
        source_format=structure.source_format,
        source_name=structure.source_name,
    )


def strip_hydrogens_from_residue(residue: Residue) -> Residue:
    """Return a residue with hydrogen atoms removed."""

    if not any(atom.element == "H" for atom in residue.atoms):
        return residue

    return replace(
        residue,
        atoms=tuple(atom for atom in residue.atoms if atom.element != "H"),
    )


def normalize_residue_inputs(
    residue: Residue,
    component_library: ComponentLibrary,
    repairs: list[RepairEvent],
) -> Residue:
    """Normalize supported residue aliases before repair."""

    normalized_component_id = component_library.normalize_component_id(
        residue.component_id
    )
    normalized_residue = residue
    if normalized_component_id != residue.component_id:
        normalized_residue = replace(residue, component_id=normalized_component_id)
        repairs.append(
            RepairEvent(
                kind=RepairEventKind.COMPONENT_NORMALIZED,
                residue_id=residue.residue_id,
                component_id=normalized_component_id,
                atom_names=(),
                details=(
                    f"normalized component {residue.component_id} -> "
                    f"{normalized_component_id}"
                ),
            )
        )

    if (
        normalized_component_id == "ILE"
        and normalized_residue.has_atom("CD")
        and not normalized_residue.has_atom("CD1")
    ):
        normalized_residue = replace(
            normalized_residue,
            atoms=tuple(
                rename_atom(atom, ILE_ATOM_ALIASES.get(atom.name, atom.name))
                for atom in normalized_residue.atoms
            ),
        )

    return normalized_residue


def normalized_reference_residue_by_id(
    reference_structure: ProteinStructure | None,
    *,
    chain_id: str,
    component_library: ComponentLibrary,
) -> dict[ResidueId, Residue]:
    """Return normalized reference residues for one chain keyed by residue id."""

    if reference_structure is None or not reference_structure.has_chain(chain_id):
        return {}

    normalized_residue_by_id: dict[ResidueId, Residue] = {}
    for residue in reference_structure.chain(chain_id).residues:
        normalized_residue = normalize_residue_inputs(
            residue,
            component_library,
            repairs=[],
        )
        normalized_residue_by_id[normalized_residue.residue_id] = normalized_residue

    return normalized_residue_by_id


def repair_residue(
    *,
    residue: Residue,
    previous_residue: Residue,
    next_residue: Residue,
    missing_atoms: tuple[str, ...],
    template: ResidueTemplate,
    reference_residue: Residue | None,
) -> tuple[Residue, tuple[str, ...]]:
    """Repair a supported residue and return added atom names."""

    if not missing_atoms:
        return residue, ()

    original_atom_names = frozenset(atom.name for atom in residue.atoms)
    guided_residue = apply_reference_guidance(
        residue=residue,
        missing_atoms=missing_atoms,
        template=template,
        reference_residue=reference_residue,
    )
    remaining_missing_atoms = missing_atoms_for_repair(guided_residue, template)
    if not remaining_missing_atoms:
        ordered_residue = reorder_residue_to_template(guided_residue, template)
        added_atoms = tuple(
            atom.name
            for atom in ordered_residue.atoms
            if atom.name not in original_atom_names
        )
        return ordered_residue, added_atoms

    repair_payload = call_repair_engine(
        residue=guided_residue,
        previous_residue=previous_residue,
        next_residue=next_residue,
        missing_atoms=list(remaining_missing_atoms),
        template=template,
    )
    repaired_residue = residue_from_payload(guided_residue, repair_payload)
    added_atoms = tuple(
        atom.name
        for atom in repaired_residue.atoms
        if atom.name not in original_atom_names
    )
    return repaired_residue, added_atoms


def apply_reference_guidance(
    *,
    residue: Residue,
    missing_atoms: tuple[str, ...],
    template: ResidueTemplate,
    reference_residue: Residue | None,
) -> Residue:
    """Fill missing sidechain atoms from an optional packed reference residue."""

    if reference_residue is None:
        return residue

    if reference_residue.component_id != residue.component_id:
        return residue

    guided_atoms = tuple(
        reference_residue.atom(atom_name)
        for atom_name in missing_atoms
        if atom_name not in BACKBONE_ATOM_NAMES
        and atom_name not in TERMINAL_EXCLUDED_ATOMS
        and reference_residue.has_atom(atom_name)
    )
    if not guided_atoms:
        return residue

    del template
    return residue.with_atoms(guided_atoms)


def reorder_residue_to_template(
    residue: Residue, template: ResidueTemplate
) -> Residue:
    """Return a residue with atoms projected into template heavy-atom order."""

    heavy_atom_semantics = template.heavy_atom_semantics
    if heavy_atom_semantics is None:
        return residue

    payload = OrderedAtomPayload(
        atom_names=list(heavy_atom_semantics.atom_order),
        atom_coordinates=[
            atom_to_vector(residue.atom(atom_name))
            for atom_name in heavy_atom_semantics.atom_order
        ],
    )
    return residue_from_payload(residue, payload)


def call_repair_engine(
    *,
    residue: Residue,
    previous_residue: Residue,
    next_residue: Residue,
    missing_atoms: list[str],
    template: ResidueTemplate,
) -> OrderedAtomPayload:
    """Use the internal geometry engine for one residue."""

    heavy_atom_semantics = template.heavy_atom_semantics
    if heavy_atom_semantics is None:
        raise ValueError("heavy repair requires heavy-atom semantics")

    return repair_residue_payload(
        payload=ordered_atom_payload(residue),
        missing_atoms=missing_atoms,
        context=HeavyRepairContext(
            next_residue_coordinates=residue_coordinates(next_residue),
            psi_points=psi_points_for_repair(previous_residue, residue),
        ),
        semantics=heavy_atom_semantics,
    )


def add_c_terminal_oxt(residue: Residue) -> Residue:
    """Add a C-terminal OXT atom using the internal geometry engine."""

    payload = ordered_atom_payload(residue)
    oxt_coordinates = c_terminal_oxt(payload)
    repaired_payload = OrderedAtomPayload(
        atom_names=payload.atom_names + ["OXT"],
        atom_coordinates=payload.atom_coordinates + [oxt_coordinates],
    )
    return residue_from_payload(residue, repaired_payload)


def residue_from_payload(
    residue: Residue,
    payload: OrderedAtomPayload,
) -> Residue:
    """Project an ordered atom payload back into the canonical residue."""

    original_atoms = {atom.name: atom for atom in residue.atoms}
    repaired_atoms: list[Atom] = []
    for atom_name, coordinates in zip(
        payload.atom_names,
        payload.atom_coordinates,
        strict=True,
    ):
        existing_atom = original_atoms.get(atom_name)
        if existing_atom is not None:
            repaired_atoms.append(
                existing_atom.with_position(vector_to_position(coordinates))
            )
            continue

        template_atom = default_atom_for_new_name(atom_name, original_atoms)
        repaired_atoms.append(
            replace(
                template_atom,
                name=atom_name,
                element=infer_element(atom_name),
                position=vector_to_position(coordinates),
            )
        )

    return replace(residue, atoms=tuple(repaired_atoms))


def default_atom_for_new_name(
    atom_name: str,
    original_atoms: Mapping[str, Atom],
) -> Atom:
    """Select a source atom template when creating a new repaired atom."""

    del atom_name
    return next(iter(original_atoms.values()))


def missing_atoms_for_repair(
    residue: Residue, template: ResidueTemplate
) -> tuple[str, ...]:
    """Return repair-target atoms excluding terminal OXT handling."""

    return template.missing_atom_names(
        tuple(atom.name for atom in residue.atoms),
        exclude_atom_names=TERMINAL_EXCLUDED_ATOMS,
    )


def residue_coordinates(residue: Residue) -> list[list[float]]:
    """Return residue coordinates as mutable numeric lists."""

    return [
        [atom.position.x, atom.position.y, atom.position.z] for atom in residue.atoms
    ]


def ordered_atom_payload(residue: Residue) -> OrderedAtomPayload:
    """Project a canonical residue into the ordered repair payload."""

    atom_names: list[str] = []
    atom_coordinates: list[list[float]] = []
    for atom in residue.atoms:
        atom_names.append(atom.name)
        atom_coordinates.append([atom.position.x, atom.position.y, atom.position.z])

    return OrderedAtomPayload(
        atom_names=atom_names,
        atom_coordinates=atom_coordinates,
    )


def psi_points_for_repair(
    previous_residue: Residue,
    residue: Residue,
) -> list[list[float]]:
    """Build the four coordinates required by the legacy backbone-O kernel."""

    return [
        atom_to_vector(previous_residue.atom("N")),
        atom_to_vector(previous_residue.atom("CA")),
        atom_to_vector(previous_residue.atom("C")),
        atom_to_vector(residue.atom("N")),
    ]


def atom_to_vector(atom: Atom) -> list[float]:
    """Convert an atom position to a mutable three-float vector."""

    return [atom.position.x, atom.position.y, atom.position.z]


def vector_to_position(coordinates: list[float]) -> Vec3:
    """Convert a numeric vector back into the canonical position object."""

    return Vec3(
        x=float(coordinates[0]),
        y=float(coordinates[1]),
        z=float(coordinates[2]),
    )


def rename_atom(atom: Atom, atom_name: str) -> Atom:
    """Return a copy of an atom with a new canonical name."""

    return replace(atom, name=atom_name)


def infer_element(atom_name: str) -> str:
    """Infer the element symbol for a standard-residue repaired atom."""

    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        raise ValueError(f"cannot infer element for atom name {atom_name!r}")

    return letters[0]
