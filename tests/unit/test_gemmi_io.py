"""Focused tests for the gemmi-backed I/O boundary."""

from pathlib import Path

import pytest

pytest.importorskip("gemmi")

from tests.support.structure_summary import summarize_structure

from pras.io import read_structure, read_structure_string, write_structure_string
from pras.model import (
    Atom,
    FileFormat,
    LigandPolicy,
    MutationPolicy,
    OccupancyPolicy,
    ProteinStructure,
    Residue,
    ResidueId,
    Vec3,
)
from pras.process import ProcessOptions


def test_read_structure_string_resolves_atom_altloc_by_highest_occupancy() -> None:
    """Altloc sites should normalize to one atom by occupancy policy."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" CA ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    y=2.0,
                    z=3.0,
                    occupancy=0.30,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    altloc="B",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=4.0,
                    y=5.0,
                    z=6.0,
                    occupancy=0.70,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    atom = structure.chain("A").residues[0].atom("CA")

    assert atom.altloc == "B"
    assert atom.position == Vec3(x=4.0, y=5.0, z=6.0)


def test_read_structure_string_can_select_lowest_occupancy_atom_site() -> None:
    """Occupancy policy should affect atom-site normalization."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" CA ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    y=2.0,
                    z=3.0,
                    occupancy=0.30,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    altloc="B",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=4.0,
                    y=5.0,
                    z=6.0,
                    occupancy=0.70,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        options=ProcessOptions(occupancy_policy=OccupancyPolicy.LOWEST),
    )

    atom = structure.chain("A").residues[0].atom("CA")

    assert atom.altloc == "A"
    assert atom.position == Vec3(x=1.0, y=2.0, z=3.0)


def test_read_structure_string_can_select_residue_variant_by_mutation_policy() -> None:
    """Duplicate residue ids should normalize by residue occupancy score."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    y=2.0,
                    z=3.0,
                    occupancy=0.80,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=1.5,
                    y=2.5,
                    z=3.5,
                    occupancy=0.80,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" N  ",
                    altloc="B",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=4.0,
                    y=5.0,
                    z=6.0,
                    occupancy=0.20,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" CA ",
                    altloc="B",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=4.5,
                    y=5.5,
                    z=6.5,
                    occupancy=0.20,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )
    lowest_structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    y=2.0,
                    z=3.0,
                    occupancy=0.80,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=1.5,
                    y=2.5,
                    z=3.5,
                    occupancy=0.80,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" N  ",
                    altloc="B",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=4.0,
                    y=5.0,
                    z=6.0,
                    occupancy=0.20,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" CA ",
                    altloc="B",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=4.5,
                    y=5.5,
                    z=6.5,
                    occupancy=0.20,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        options=ProcessOptions(
            mutation_policy=MutationPolicy.LOWEST_OCCUPANCY,
        ),
    )

    assert structure.chain("A").residues[0].component_id == "ALA"
    assert lowest_structure.chain("A").residues[0].component_id == "GLY"


def test_read_structure_string_can_filter_chains_and_drop_ligands() -> None:
    """Boundary options should normalize chain and ligand selection at ingress."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="B",
                    residue_seq=2,
                    x=4.0,
                    y=5.0,
                    z=6.0,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=3,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="FAD",
                    chain_id="B",
                    residue_seq=3,
                    x=7.0,
                    y=8.0,
                    z=9.0,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        options=ProcessOptions(
            selected_chain_ids=("B",),
            ligand_policy=LigandPolicy.DROP,
        ),
    )

    assert structure.chain_ids() == ("B",)
    assert structure.ligands == ()


def test_read_structure_string_keeps_non_water_ligands_only() -> None:
    """Ligand retention should exclude water from the ligand bucket."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" O  ",
                    residue_name="HOH",
                    chain_id="A",
                    residue_seq=2,
                    x=4.0,
                    y=5.0,
                    z=6.0,
                    element="O",
                ),
                build_pdb_atom_line(
                    serial=3,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="FAD",
                    chain_id="A",
                    residue_seq=3,
                    x=7.0,
                    y=8.0,
                    z=9.0,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        options=ProcessOptions(ligand_policy=LigandPolicy.KEEP),
    )

    assert structure.chain_ids() == ("A",)
    assert tuple(residue.component_id for residue in structure.ligands) == ("FAD",)


def test_write_structure_string_roundtrips_pdb_and_mmcif() -> None:
    """PDB and mmCIF serialization should roundtrip to the same semantics."""

    structure = build_canonical_structure()

    pdb_roundtrip = read_structure_string(
        write_structure_string(structure, FileFormat.PDB),
        FileFormat.PDB,
    )
    mmcif_roundtrip = read_structure_string(
        write_structure_string(structure, FileFormat.MMCIF),
        FileFormat.MMCIF,
    )

    assert summarize_structure(pdb_roundtrip) == summarize_structure(structure)
    assert summarize_structure(mmcif_roundtrip) == summarize_structure(structure)


def test_read_structure_infers_format_from_path_suffix(tmp_path: Path) -> None:
    """Path-based reading should infer PDB and mmCIF formats from suffixes."""

    structure = build_canonical_structure()
    pdb_path = tmp_path / "fixture.pdb"
    cif_path = tmp_path / "fixture.cif"
    pdb_path.write_text(
        write_structure_string(structure, FileFormat.PDB),
        encoding="utf-8",
    )
    cif_path.write_text(
        write_structure_string(structure, FileFormat.MMCIF),
        encoding="utf-8",
    )

    pdb_structure = read_structure(pdb_path)
    cif_structure = read_structure(cif_path)

    assert summarize_structure(pdb_structure) == summarize_structure(structure)
    assert summarize_structure(cif_structure) == summarize_structure(structure)


def build_canonical_structure() -> ProteinStructure:
    """Build a small canonical structure for I/O roundtrip tests."""

    return ProteinStructure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),
                    build_residue("SER", "A", 2, ("N", "CA", "C", "O", "CB")),
                ),
            ),
            build_chain(
                "B",
                (
                    build_residue("TYR", "B", 10, ("N", "CA", "C", "O", "CB", "CG")),
                ),
            ),
        ),
        ligands=(
            build_residue("FAD", "B", 99, ("C1", "N1", "O1"), is_hetero=True),
        ),
        source_format=FileFormat.PDB,
        source_name="fixture",
    )


def build_chain(chain_id: str, residues: tuple[Residue, ...]):
    """Build a canonical chain for roundtrip tests."""

    from pras.model import Chain

    return Chain(chain_id=chain_id, residues=residues)


def build_residue(
    component_id: str,
    chain_id: str,
    seq_num: int,
    atom_names: tuple[str, ...],
    *,
    is_hetero: bool = False,
) -> Residue:
    """Build a canonical residue for roundtrip tests."""

    atoms = tuple(
        build_atom(atom_name, atom_index)
        for atom_index, atom_name in enumerate(atom_names, start=1)
    )
    return Residue(
        component_id=component_id,
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atoms=atoms,
        is_hetero=is_hetero,
    )


def build_atom(atom_name: str, atom_index: int) -> Atom:
    """Build a canonical atom with deterministic coordinates."""

    return Atom(
        name=atom_name,
        element=infer_element(atom_name),
        position=Vec3(
            x=float(atom_index),
            y=float(atom_index) + 0.5,
            z=float(atom_index) + 1.0,
        ),
        occupancy=1.0,
        b_factor=20.0,
    )


def infer_element(atom_name: str) -> str:
    """Infer a simple test element from an atom name."""

    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        raise ValueError(f"atom_name must contain at least one letter: {atom_name}")

    return letters[0]


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
    """Build one fixed-width PDB atom record for gemmi reader tests."""

    return (
        f"{record_name:<6}{serial:>5} {atom_name}{altloc}{residue_name:>3} "
        f"{chain_id}{residue_seq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}"
        f"          {element:>2}  "
    )
