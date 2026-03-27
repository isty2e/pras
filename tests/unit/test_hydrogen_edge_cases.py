"""Adversarial edge cases for hydrogen placement."""

from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.structure_summary import summarize_structure

from pras.io import read_structure, read_structure_string
from pras.model import (
    Chain,
    FileFormat,
    LigandPolicy,
    MutationPolicy,
    ProteinStructure,
    Residue,
    ResidueId,
    Vec3,
)
from pras.process import ProcessOptions
from pras.repair import add_hydrogens
from pras.repair.geometry import RotatableHydrogenSearch, optimize_rotatable_hydrogen
from pras.repair.payloads import ResiduePayload, RotatableHydrogenEnvironment


def test_single_residue_chain_gets_only_n_terminal_backbone_hydrogens() -> None:
    """A single-residue chain should not receive a propagated backbone H atom."""

    structure = structure_from_tokens(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        ("A:19",),
    )

    result = add_hydrogens(structure)
    residue = result.structure.chain("A").residues[0]

    assert {"H1", "H2", "H3"}.issubset(set(residue.atom_names()))
    assert "H" not in residue.atom_names()


def test_proline_does_not_receive_backbone_hydrogen_from_previous_residue() -> None:
    """A residue preceding proline should not add the backbone H onto proline."""

    structure = structure_from_tokens(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        ("A:10", "A:11"),
    )

    result = add_hydrogens(structure)
    proline = result.structure.chain("A").residues[1]

    assert proline.component_id == "PRO"
    assert "H" not in proline.atom_names()


@pytest.mark.parametrize(
    ("distance", "expect_hg"),
    (
        pytest.param(3.0, False, id="threshold-bond"),
        pytest.param(3.01, True, id="outside-threshold"),
    ),
)
def test_cysteine_hg_depends_on_disulfide_distance_threshold(
    distance: float, expect_hg: bool
) -> None:
    """Cysteine HG placement should flip exactly at the disulfide cutoff."""

    structure = disulfide_threshold_structure(distance)

    result = add_hydrogens(structure)
    first_residue, second_residue = result.structure.chain("A").residues

    assert first_residue.has_atom("HG") is expect_hg
    assert second_residue.has_atom("HG") is expect_hg


@pytest.mark.parametrize(
    ("count", "expected_protonated"),
    (
        pytest.param(4, 0, id="four-his-no-protonation"),
        pytest.param(5, 1, id="five-his-one-protonated"),
    ),
)
def test_histidine_protonation_threshold_is_deterministic(
    count: int, expected_protonated: int
) -> None:
    """The 20%-of-HIS rule should switch on only once the fifth HIS appears."""

    structure = structure_from_tokens(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        ("A:41", "A:93", "A:102", "A:106", "A:124")[:count],
    )

    result = add_hydrogens(structure, protonate_histidines=True)
    residues = result.structure.chain("A").residues
    protonated = [residue for residue in residues if residue.has_atom("HD1")]

    assert len(protonated) == expected_protonated
    if expected_protonated:
        assert protonated[0].residue_id == residues[0].residue_id


def test_insertion_code_survives_class6_hydrogen_placement() -> None:
    """Insertion codes should survive contextual hydrogen placement unchanged."""

    structure = structure_from_tokens(
        Path("tests/fixtures/pdb/1aho.pdb"),
        ("A:40",),
    )
    original_residue = structure.chain("A").residues[0]
    insertion_residue = replace(
        original_residue,
        residue_id=ResidueId(chain_id="A", seq_num=40, insertion_code="A"),
    )
    structure = ProteinStructure(
        chains=(Chain(chain_id="A", residues=(insertion_residue,)),),
        ligands=(),
        source_format=structure.source_format,
        source_name=structure.source_name,
    )

    result = add_hydrogens(structure)
    residue = result.structure.chain("A").residues[0]

    assert residue.residue_id.insertion_code == "A"
    assert residue.has_atom("HG")


def test_rotatable_hydrogen_falls_back_to_initial_position_on_legacy_index_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy duplicate residue numbers should fall back to the initial hydrogen."""

    candidate_hydrogens = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    energy_calls = iter((5.0, 6.0, 1.0, 2.0))

    monkeypatch.setattr(
        "pras.repair.geometry.rotated_hydrogen_candidates",
        lambda search: candidate_hydrogens,
    )
    monkeypatch.setattr(
        "pras.repair.geometry.hydrogen_potential_energy",
        lambda *args, **kwargs: next(energy_calls),
    )

    result = optimize_rotatable_hydrogen(
        residue_number="7",
        environments=(
            RotatableHydrogenEnvironment(
                residue_number="7",
                atom_x=(1.0,),
                atom_y=(0.0,),
                atom_z=(0.0,),
                charges=(0.0,),
                sigmas_nm=(0.0,),
                epsilons_kj_mol=(0.0,),
            ),
            RotatableHydrogenEnvironment(
                residue_number="7",
                atom_x=(0.0,),
                atom_y=(1.0,),
                atom_z=(0.0,),
                charges=(0.0,),
                sigmas_nm=(0.0,),
                epsilons_kj_mol=(0.0,),
            ),
        ),
        search=RotatableHydrogenSearch(
            outer_anchor=[0.0, 0.0, 0.0],
            inner_anchor=[0.0, 0.0, 0.0],
            donor=[0.0, 0.0, 0.0],
            hydrogen=[9.0, 9.0, 9.0],
            build_bond_length=1.0,
            reproject_bond_length=1.0,
            dihedral=0.0,
            partial_charge=0.0,
            sigma=0.0,
            epsilon=0.0,
        ),
    )

    assert result == [9.0, 9.0, 9.0]


def test_residue_payload_exposes_legacy_backbone_hydrogen_anchor_order() -> None:
    """Backbone-H propagation should preserve the legacy positional anchor quirk."""

    payload = ResiduePayload(
        residue_label="SER2",
        atom_names=["N", "C", "O", "CA", "CB", "OG"],
        atom_coordinates=[
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
        ],
    )

    alpha_anchor, nitrogen_anchor = payload.legacy_backbone_hydrogen_anchors()

    assert alpha_anchor == [2.0, 2.0, 2.0]
    assert nitrogen_anchor == [1.0, 1.0, 1.0]


def test_ligand_keep_with_water_does_not_pollute_hydrogenated_structure() -> None:
    """Ligand retention should keep non-water ligands and still hydrogenate chains."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    y=1.0,
                    z=1.0,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    y=1.5,
                    z=1.0,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" C  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=3.0,
                    y=1.0,
                    z=1.5,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" O  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=3.8,
                    y=1.2,
                    z=2.3,
                    element="O",
                ),
                build_pdb_atom_line(
                    serial=5,
                    record_name="HETATM",
                    atom_name=" O  ",
                    residue_name="HOH",
                    chain_id="A",
                    residue_seq=2,
                    x=6.0,
                    y=6.0,
                    z=6.0,
                    element="O",
                ),
                build_pdb_atom_line(
                    serial=6,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="FAD",
                    chain_id="A",
                    residue_seq=3,
                    x=7.0,
                    y=7.0,
                    z=7.0,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        options=ProcessOptions(ligand_policy=LigandPolicy.KEEP),
    )

    result = add_hydrogens(structure)
    residue = result.structure.chain("A").residues[0]

    assert tuple(ligand.component_id for ligand in result.structure.ligands) == ("FAD",)
    assert {"H1", "H2", "H3"}.issubset(set(residue.atom_names()))


def test_hydrogen_placement_is_stable_on_already_hydrogenated_input() -> None:
    """Re-running hydrogen placement should not drift the semantic structure."""

    structure = read_structure(
        Path("tests/fixtures/pdb/1aho.pdb"),
        options=ProcessOptions(),
    )

    first_pass = add_hydrogens(structure)
    second_pass = add_hydrogens(first_pass.structure)

    assert summarize_structure(second_pass.structure) == summarize_structure(
        first_pass.structure
    )


def test_unsupported_component_skips_hydrogenation_for_that_chain() -> None:
    """A mixed chain with unsupported chemistry should remain unhydrogenated."""

    structure = structure_from_tokens(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        ("A:19", "A:20"),
    )
    first_residue, second_residue = structure.chain("A").residues
    unsupported_residue = replace(second_residue, component_id="MSE")
    structure = ProteinStructure(
        chains=(Chain(chain_id="A", residues=(first_residue, unsupported_residue)),),
        ligands=(),
        source_format=structure.source_format,
        source_name=structure.source_name,
    )

    result = add_hydrogens(structure)
    first_after, second_after = result.structure.chain("A").residues

    assert "H1" not in first_after.atom_names()
    assert "H" not in second_after.atom_names()
    assert result.has_warnings()


def test_unsupported_component_isolation_is_per_chain_not_global() -> None:
    """An unsupported residue in one chain should not block another chain."""

    supported = structure_from_tokens(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        ("A:19",),
    ).chain("A")
    unsupported_source = structure_from_tokens(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        ("A:20",),
    ).chain("A").residues[0]
    unsupported_residue = replace(
        unsupported_source,
        component_id="MSE",
        residue_id=ResidueId(
            chain_id="B",
            seq_num=unsupported_source.residue_id.seq_num,
        ),
    )
    structure = ProteinStructure(
        chains=(
            supported,
            Chain(chain_id="B", residues=(unsupported_residue,)),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="mixed-chain",
    )

    result = add_hydrogens(structure)
    supported_after = result.structure.chain("A").residues[0]
    unsupported_after = result.structure.chain("B").residues[0]

    assert supported_after.has_atom("H1")
    assert not unsupported_after.has_atom("H1")
    assert result.has_warnings()


def test_blank_chain_id_survives_hydrogenation() -> None:
    """The normalized default chain id should survive the full repair pipeline."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id=" ",
                    residue_seq=1,
                    x=1.0,
                    y=1.0,
                    z=1.0,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    residue_name="GLY",
                    chain_id=" ",
                    residue_seq=1,
                    x=2.0,
                    y=1.5,
                    z=1.0,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" C  ",
                    residue_name="GLY",
                    chain_id=" ",
                    residue_seq=1,
                    x=3.0,
                    y=1.0,
                    z=1.5,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" O  ",
                    residue_name="GLY",
                    chain_id=" ",
                    residue_seq=1,
                    x=3.8,
                    y=1.2,
                    z=2.3,
                    element="O",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    result = add_hydrogens(structure)

    assert result.structure.chain_ids() == ("_",)
    assert result.structure.chain("_").residues[0].has_atom("H1")


def test_ligand_only_input_remains_stable_without_polymer_chains() -> None:
    """A ligand-only input should not crash or manufacture polymer chains."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="FAD",
                    chain_id="A",
                    residue_seq=1,
                    x=7.0,
                    y=7.0,
                    z=7.0,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        options=ProcessOptions(ligand_policy=LigandPolicy.KEEP),
    )

    result = add_hydrogens(structure)

    assert result.structure.chains == ()
    assert tuple(ligand.component_id for ligand in result.structure.ligands) == ("FAD",)


def test_terminal_oxt_is_added_before_hydrogenation() -> None:
    """Hydrogen placement should preserve the heavy-repair OXT terminal fix."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    y=1.0,
                    z=1.0,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    y=1.5,
                    z=1.0,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" C  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=3.0,
                    y=1.0,
                    z=1.5,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" O  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=3.8,
                    y=1.2,
                    z=2.3,
                    element="O",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    result = add_hydrogens(structure)

    assert result.structure.chain("A").residues[0].has_atom("OXT")


def test_mutation_policy_controls_hydrogenated_component_choice() -> None:
    """Mutation normalization should feed the chosen residue into hydrogenation."""

    pdb_text = build_pdb_text(
        [
            build_pdb_atom_line(
                serial=1,
                atom_name=" N  ",
                residue_name="ALA",
                chain_id="A",
                residue_seq=1,
                x=1.0,
                y=1.0,
                z=1.0,
                occupancy=0.80,
                element="N",
            ),
            build_pdb_atom_line(
                serial=2,
                atom_name=" CA ",
                residue_name="ALA",
                chain_id="A",
                residue_seq=1,
                x=2.0,
                y=1.5,
                z=1.0,
                occupancy=0.80,
                element="C",
            ),
            build_pdb_atom_line(
                serial=3,
                atom_name=" C  ",
                residue_name="ALA",
                chain_id="A",
                residue_seq=1,
                x=3.0,
                y=1.0,
                z=1.5,
                occupancy=0.80,
                element="C",
            ),
            build_pdb_atom_line(
                serial=4,
                atom_name=" O  ",
                residue_name="ALA",
                chain_id="A",
                residue_seq=1,
                x=3.8,
                y=1.2,
                z=2.3,
                occupancy=0.80,
                element="O",
            ),
            build_pdb_atom_line(
                serial=5,
                atom_name=" CB ",
                residue_name="ALA",
                chain_id="A",
                residue_seq=1,
                x=2.0,
                y=2.6,
                z=0.0,
                occupancy=0.80,
                element="C",
            ),
            build_pdb_atom_line(
                serial=6,
                atom_name=" N  ",
                residue_name="GLY",
                chain_id="A",
                residue_seq=1,
                x=1.1,
                y=1.1,
                z=1.2,
                occupancy=0.20,
                element="N",
            ),
            build_pdb_atom_line(
                serial=7,
                atom_name=" CA ",
                residue_name="GLY",
                chain_id="A",
                residue_seq=1,
                x=2.1,
                y=1.6,
                z=1.1,
                occupancy=0.20,
                element="C",
            ),
            build_pdb_atom_line(
                serial=8,
                atom_name=" C  ",
                residue_name="GLY",
                chain_id="A",
                residue_seq=1,
                x=3.1,
                y=1.1,
                z=1.6,
                occupancy=0.20,
                element="C",
            ),
            build_pdb_atom_line(
                serial=9,
                atom_name=" O  ",
                residue_name="GLY",
                chain_id="A",
                residue_seq=1,
                x=3.9,
                y=1.3,
                z=2.4,
                occupancy=0.20,
                element="O",
            ),
            "END",
        ]
    )

    highest = add_hydrogens(read_structure_string(pdb_text, FileFormat.PDB))
    lowest = add_hydrogens(
        read_structure_string(
            pdb_text,
            FileFormat.PDB,
            options=ProcessOptions(
                mutation_policy=MutationPolicy.LOWEST_OCCUPANCY
            ),
        )
    )

    highest_residue = highest.structure.chain("A").residues[0]
    lowest_residue = lowest.structure.chain("A").residues[0]

    assert highest_residue.component_id == "ALA"
    assert "HB1" in highest_residue.atom_names()
    assert lowest_residue.component_id == "GLY"
    assert {"HA1", "HA2"}.issubset(set(lowest_residue.atom_names()))


def test_altloc_selection_survives_hydrogenation_without_duplicate_atoms() -> None:
    """Hydrogen placement should preserve the chosen heavy-atom altloc geometry."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    y=1.0,
                    z=1.0,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    y=1.5,
                    z=1.0,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" C  ",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=3.0,
                    y=1.0,
                    z=1.5,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" O  ",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=3.8,
                    y=1.2,
                    z=2.3,
                    element="O",
                ),
                build_pdb_atom_line(
                    serial=5,
                    atom_name=" CB ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    y=2.6,
                    z=0.0,
                    occupancy=0.30,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=6,
                    atom_name=" CB ",
                    altloc="B",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=6.0,
                    y=6.0,
                    z=6.0,
                    occupancy=0.70,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    result = add_hydrogens(structure)
    residue = result.structure.chain("A").residues[0]
    cb = residue.atom("CB")

    assert cb.position == Vec3(x=6.0, y=6.0, z=6.0)
    assert len(residue.atom_names()) == len(set(residue.atom_names()))


def test_histidine_alias_is_normalized_before_hydrogenation() -> None:
    """Histidine aliases should normalize into canonical HIS before hydrogenation."""

    structure = structure_from_tokens(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        ("A:41",),
    )
    residue = replace(structure.chain("A").residues[0], component_id="HSE")
    structure = ProteinStructure(
        chains=(Chain(chain_id="A", residues=(residue,)),),
        ligands=(),
        source_format=structure.source_format,
        source_name=structure.source_name,
    )

    result = add_hydrogens(structure)
    normalized = result.structure.chain("A").residues[0]

    assert normalized.component_id == "HIS"
    assert {"HD2", "HE1", "HE2"}.issubset(set(normalized.atom_names()))


def test_empty_structure_survives_hydrogenation() -> None:
    """An empty structure should pass through hydrogen placement unchanged."""

    structure = read_structure_string(build_pdb_text(["END"]), FileFormat.PDB)

    result = add_hydrogens(structure)

    assert result.structure.chains == ()
    assert result.structure.ligands == ()


def test_chain_selection_and_ligand_keep_stay_aligned_through_hydrogenation() -> None:
    """Selected chains and their ligands should stay aligned after hydrogenation."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    y=1.0,
                    z=1.0,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    y=1.5,
                    z=1.0,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" C  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=3.0,
                    y=1.0,
                    z=1.5,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" O  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=3.8,
                    y=1.2,
                    z=2.3,
                    element="O",
                ),
                build_pdb_atom_line(
                    serial=5,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="B",
                    residue_seq=1,
                    x=10.0,
                    y=1.0,
                    z=1.0,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=6,
                    atom_name=" CA ",
                    residue_name="GLY",
                    chain_id="B",
                    residue_seq=1,
                    x=11.0,
                    y=1.5,
                    z=1.0,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=7,
                    atom_name=" C  ",
                    residue_name="GLY",
                    chain_id="B",
                    residue_seq=1,
                    x=12.0,
                    y=1.0,
                    z=1.5,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=8,
                    atom_name=" O  ",
                    residue_name="GLY",
                    chain_id="B",
                    residue_seq=1,
                    x=12.8,
                    y=1.2,
                    z=2.3,
                    element="O",
                ),
                build_pdb_atom_line(
                    serial=9,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="FAD",
                    chain_id="B",
                    residue_seq=2,
                    x=14.0,
                    y=7.0,
                    z=7.0,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        options=ProcessOptions(
            selected_chain_ids=("B",),
            ligand_policy=LigandPolicy.KEEP,
        ),
    )

    result = add_hydrogens(structure)

    assert result.structure.chain_ids() == ("B",)
    assert result.structure.chain("B").residues[0].has_atom("H1")
    assert tuple(ligand.component_id for ligand in result.structure.ligands) == ("FAD",)


def structure_from_tokens(
    path: Path, residue_tokens: tuple[str, ...]
) -> ProteinStructure:
    """Extract a minimal canonical structure from selected real residues."""

    structure = read_structure(path, options=ProcessOptions())
    chain_by_token = {
        residue.residue_id.display_token(): residue
        for residue in structure.iter_residues()
    }
    residues = tuple(chain_by_token[token] for token in residue_tokens)
    chain_id = residues[0].residue_id.chain_id
    return ProteinStructure(
        chains=(Chain(chain_id=chain_id, residues=residues),),
        ligands=(),
        source_format=structure.source_format,
        source_name=structure.source_name,
    )


def disulfide_threshold_structure(distance: float) -> ProteinStructure:
    """Build a two-cysteine structure with a chosen SG-SG distance."""

    structure = structure_from_tokens(
        Path("tests/fixtures/pdb/1aho.pdb"),
        ("A:12", "A:63"),
    )
    first_residue, second_residue = structure.chain("A").residues
    first_sg = first_residue.atom("SG").position
    second_sg = second_residue.atom("SG").position
    target_sg = Vec3(
        x=first_sg.x + distance,
        y=first_sg.y,
        z=first_sg.z,
    )
    shifted_second = translate_residue(
        second_residue,
        dx=target_sg.x - second_sg.x,
        dy=target_sg.y - second_sg.y,
        dz=target_sg.z - second_sg.z,
    )
    return ProteinStructure(
        chains=(Chain(chain_id="A", residues=(first_residue, shifted_second)),),
        ligands=(),
        source_format=structure.source_format,
        source_name=structure.source_name,
    )


def translate_residue(residue: Residue, *, dx: float, dy: float, dz: float) -> Residue:
    """Translate all residue atom coordinates by a fixed offset."""

    translated_atoms = tuple(
        atom.with_position(atom.position.with_offset(dx, dy, dz))
        for atom in residue.atoms
    )
    return replace(residue, atoms=translated_atoms)


def build_pdb_text(lines: list[str]) -> str:
    """Join fixed-width PDB records into a text payload."""

    return "\n".join(lines) + "\n"


def build_pdb_atom_line(
    *,
    serial: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_seq: int,
    record_name: str = "ATOM",
    altloc: str = " ",
    x: float = 1.0,
    y: float = 2.0,
    z: float = 3.0,
    occupancy: float = 1.0,
    b_factor: float = 20.0,
    element: str = "",
) -> str:
    """Build one fixed-width PDB atom record for ingress edge tests."""

    return (
        f"{record_name:<6}{serial:>5} {atom_name}{altloc}{residue_name:>3} "
        f"{chain_id}{residue_seq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}"
        f"          {element:>2}  "
    )
