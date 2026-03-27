"""Unit tests for the redesigned PRAS domain model."""

import numpy as np

from pras.chemistry import (
    ChemicalComponentDefinition,
    ComponentCapability,
    ComponentLibrary,
    ResidueTemplate,
    build_standard_component_library,
)
from pras.model import (
    AnalysisBundle,
    AnalysisKind,
    Atom,
    Chain,
    FileFormat,
    HydrogenPolicy,
    IssueSeverity,
    LigandPolicy,
    MutationPolicy,
    OccupancyPolicy,
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
from pras.packing import PackingScope, PackingSpec
from pras.process import ProcessOptions


def test_vec3_supports_distance_and_array_conversion() -> None:
    left = Vec3(0.0, 0.0, 0.0)
    right = Vec3.from_iterable([1.0, 2.0, 2.0])

    assert left.distance_to(right) == 3.0
    assert np.array_equal(
        right.to_array(),
        np.asarray([1.0, 2.0, 2.0], dtype=np.float64),
    )


def test_residue_validates_against_component_definition() -> None:
    definition = ChemicalComponentDefinition(
        component_id="ALA",
        atom_names=("N", "CA", "C", "O", "CB"),
        aliases=("Ala",),
        capabilities=frozenset({ComponentCapability.TEMPLATE_REPAIR}),
    )
    residue = Residue(
        component_id="ALA",
        residue_id=ResidueId(chain_id="A", seq_num=10),
        atoms=(
            Atom("N", "N", Vec3(0.0, 0.0, 0.0)),
            Atom("CA", "C", Vec3(1.0, 0.0, 0.0)),
            Atom("C", "C", Vec3(2.0, 0.0, 0.0)),
            Atom("XX", "C", Vec3(3.0, 0.0, 0.0)),
        ),
    )

    assert residue.missing_atoms(definition) == ("O", "CB")
    assert residue.unexpected_atoms(definition) == ("XX",)

    issues = residue.validate_against(definition)

    assert len(issues) == 2
    assert {issue.severity for issue in issues} == {IssueSeverity.WARNING}


def test_residue_with_atom_replaces_existing_name() -> None:
    residue = Residue(
        component_id="GLY",
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atoms=(
            Atom("N", "N", Vec3(0.0, 0.0, 0.0)),
            Atom("CA", "C", Vec3(1.0, 0.0, 0.0)),
        ),
    )

    updated = residue.with_atom(Atom("CA", "C", Vec3(5.0, 0.0, 0.0)))

    assert updated.atom("CA").position == Vec3(5.0, 0.0, 0.0)
    assert residue.atom("CA").position == Vec3(1.0, 0.0, 0.0)


def test_component_library_normalizes_aliases() -> None:
    definition = ChemicalComponentDefinition(
        component_id="HIS",
        atom_names=("N", "CA", "C"),
        aliases=("HSD", "HSE"),
    )
    template = ResidueTemplate(definition=definition)
    library = ComponentLibrary(templates={"HIS": template})

    assert library.normalize_component_id("hsd") == "HIS"
    assert library.require("HSE") == template


def test_standard_templates_expose_semantics_and_forcefield_data() -> None:
    library = build_standard_component_library()
    histidine_template = library.require("HSE")
    serine_template = library.require("SER")

    assert histidine_template.component_id == "HIS"
    assert histidine_template.can_add_hydrogens()
    assert serine_template.can_repair_heavy_atoms()
    assert serine_template.has_forcefield_params("OG")
    assert serine_template.missing_atom_names(
        ("N", "CA", "C", "O"),
        exclude_atom_names=("OXT",),
    ) == ("CB", "OG")


def test_standard_templates_cover_heavy_repair_for_all_standard_residues() -> None:
    library = build_standard_component_library()
    standard_component_ids = (
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    )

    assert all(
        library.require(component_id).can_repair_heavy_atoms()
        for component_id in standard_component_ids
    )


def test_chain_and_structure_support_rich_navigation() -> None:
    residue_1 = Residue(
        component_id="GLY",
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atoms=(Atom("N", "N", Vec3(0.0, 0.0, 0.0)),),
    )
    residue_2 = Residue(
        component_id="ALA",
        residue_id=ResidueId(chain_id="A", seq_num=2),
        atoms=(Atom("N", "N", Vec3(1.0, 0.0, 0.0)),),
    )
    residue_3 = Residue(
        component_id="SER",
        residue_id=ResidueId(chain_id="B", seq_num=1),
        atoms=(Atom("N", "N", Vec3(2.0, 0.0, 0.0)),),
    )
    chain_a = Chain(chain_id="A", residues=(residue_1, residue_2))
    chain_b = Chain(chain_id="B", residues=(residue_3,))
    structure = ProteinStructure(
        chains=(chain_a, chain_b),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="fixture",
    )

    assert chain_a.residue_window(residue_2.residue_id, radius=1) == (
        residue_1,
        residue_2,
    )
    assert structure.select_chains(("B",)).chain_ids() == ("B",)
    assert tuple(atom.name for atom in structure.iter_atoms()) == ("N", "N", "N")


def test_process_options_normalize_selection_and_requested_analyses() -> None:
    options = ProcessOptions(
        occupancy_policy=OccupancyPolicy.HIGHEST,
        mutation_policy=MutationPolicy.HIGHEST_OCCUPANCY,
        ligand_policy=LigandPolicy.DROP,
        hydrogen_policy=HydrogenPolicy.PRESERVE,
        selected_chain_ids=("A", "A", "B"),
        sidechain_packing=PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
        analyses=frozenset({AnalysisKind.SECONDARY_STRUCTURE}),
    )

    enriched = options.with_requested_analysis(AnalysisKind.RAMACHANDRAN)

    assert options.selected_chain_ids == ("A", "B")
    assert options.selects_chain("A")
    assert not options.selects_chain("C")
    assert options.requests_sidechain_packing()
    assert enriched.requests_analysis(AnalysisKind.RAMACHANDRAN)


def test_process_result_tracks_repairs_and_issues() -> None:
    structure = ProteinStructure(chains=(), ligands=(), source_format=FileFormat.MMCIF)
    repair = RepairEvent(
        kind=RepairEventKind.HEAVY_ATOMS_ADDED,
        residue_id=ResidueId(chain_id="A", seq_num=10),
        component_id="ALA",
        atom_names=("CB",),
    )
    issue = ValidationIssue(
        kind=ValidationIssueKind.UNEXPECTED_ATOMS,
        severity=IssueSeverity.WARNING,
        message="unexpected atom",
    )
    result = ProcessResult(
        structure=structure,
        repairs=(repair,),
        issues=(issue,),
        analyses=AnalysisBundle(),
    )

    assert result.repair_count() == 1
    assert result.issue_count() == 1
    assert not result.has_errors()
    assert result.has_warnings()
