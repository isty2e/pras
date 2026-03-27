"""Unit and smoke tests for the FASPR side-chain packing backend."""

from pathlib import Path

import pytest

from pras.io import read_structure
from pras.model import (
    Atom,
    Chain,
    FileFormat,
    ProteinStructure,
    Residue,
    ResidueId,
    Vec3,
)
from pras.packing import (
    PackingMode,
    PackingRequest,
    PackingScope,
    PackingSpec,
    faspr_executable_path,
)
from pras.packing.faspr_backend import FasprPackingBackend
from pras.process import ProcessOptions

FASPR_FIXTURE_PATH = Path("tests/fixtures/pdb/1aho_faspr_input.pdb")


def test_faspr_backend_declares_expected_capabilities() -> None:
    """FASPR should advertise the expected fixed-backbone packing surface."""

    capabilities = FasprPackingBackend().capabilities()

    assert capabilities.supports_full_structure_packing
    assert capabilities.supports_local_packing
    assert capabilities.supports_partial_sequence
    assert not capabilities.supports_refinement
    assert not capabilities.supports_noncanonical_components
    assert capabilities.deterministic_given_same_inputs


def test_faspr_backend_builds_local_sequence_override_with_fake_executable(
    tmp_path: Path,
) -> None:
    """Local packing should translate mutable/fixed residues into FASPR casing."""

    log_path = tmp_path / "sequence.log"
    executable_path = tmp_path / "FASPR"
    executable_path.write_text(
        "\n".join(
            (
                "#!/bin/sh",
                "set -eu",
                'input_path=""',
                'output_path=""',
                'sequence_path=""',
                'while [ "$#" -gt 0 ]; do',
                '  case "$1" in',
                '    -i) input_path="$2"; shift 2 ;;',
                '    -o) output_path="$2"; shift 2 ;;',
                '    -s) sequence_path="$2"; shift 2 ;;',
                "    *) shift ;;",
                "  esac",
                "done",
                'cp "$input_path" "$output_path"',
                'if [ -n "$sequence_path" ]; then',
                f'  cat "$sequence_path" > "{log_path}"',
                "else",
                f'  : > "{log_path}"',
                "fi",
            )
        ),
        encoding="utf-8",
    )
    executable_path.chmod(0o755)
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")

    structure = build_test_structure()
    mutable_residue_id = structure.chain("A").residues[1].residue_id
    request = PackingRequest(
        structure=structure,
        spec=PackingSpec(
            backend_name="faspr",
            mode=PackingMode.PACK,
            scope=PackingScope.LOCAL,
            mutable_residue_ids=(mutable_residue_id,),
            target_sequence="V",
        ),
    )

    result = FasprPackingBackend(executable_path=executable_path).pack(request)

    assert log_path.read_text(encoding="utf-8").strip() == "aVy"
    assert result.backend_name == "faspr"
    assert result.changed_residue_ids == ()
    assert result.packed_structure.chain_ids() == ("A",)


def test_faspr_backend_smoke_runs_packaged_binary() -> None:
    """The packaged FASPR executable should pack a representative fixture."""

    if not FASPR_FIXTURE_PATH.exists():
        pytest.skip("FASPR input fixture is unavailable")

    try:
        faspr_executable_path()
    except FileNotFoundError:
        pytest.skip("packaged FASPR executable is unavailable")

    structure = read_structure(
        FASPR_FIXTURE_PATH,
        options=ProcessOptions(),
    )
    request = PackingRequest(
        structure=structure,
        spec=PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    result = FasprPackingBackend().pack(request)

    assert result.backend_name == "faspr"
    assert result.packed_structure.chain_ids() == structure.chain_ids()
    assert sum(len(chain.residues) for chain in result.packed_structure.chains) == sum(
        len(chain.residues) for chain in structure.chains
    )
    assert result.packed_structure.ligands == structure.ligands


def build_test_structure() -> ProteinStructure:
    """Build one small canonical structure for backend tests."""

    return ProteinStructure(
        chains=(
            Chain(
                chain_id="A",
                residues=(
                    build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),
                    build_residue("LEU", "A", 2, ("N", "CA", "C", "O", "CB", "CG")),
                    build_residue("TYR", "A", 3, ("N", "CA", "C", "O", "CB", "CG")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="faspr-backend-test",
    )


def build_residue(
    component_id: str,
    chain_id: str,
    seq_num: int,
    atom_names: tuple[str, ...],
) -> Residue:
    """Build one canonical residue for backend tests."""

    atoms = tuple(
        build_atom(atom_name, atom_index)
        for atom_index, atom_name in enumerate(atom_names, start=1)
    )
    return Residue(
        component_id=component_id,
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atoms=atoms,
    )


def build_atom(atom_name: str, atom_index: int) -> Atom:
    """Build one deterministic canonical atom for backend tests."""

    preset_positions = (
        Vec3(0.000, 0.000, 0.000),
        Vec3(1.458, 0.000, 0.000),
        Vec3(2.028, 1.417, 0.000),
        Vec3(3.235, 1.593, 0.248),
        Vec3(1.145, -0.842, 1.074),
        Vec3(2.318, -1.152, 1.556),
    )
    return Atom(
        name=atom_name,
        element=infer_element(atom_name),
        position=preset_positions[(atom_index - 1) % len(preset_positions)],
        occupancy=1.0,
        b_factor=20.0,
    )


def infer_element(atom_name: str) -> str:
    """Infer a simple element token from an atom name."""

    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        raise ValueError(f"atom_name must contain at least one letter: {atom_name}")

    return letters[0]
