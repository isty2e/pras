"""Representative regression tests for heavy-atom repair."""

from pathlib import Path

from tests.support.representative_cases import REPRESENTATIVE_CASES
from tests.support.structure_summary import summarize_structure

from pras.io import read_structure
from pras.model import FileFormat, LigandPolicy, ProteinStructure
from pras.process import ProcessOptions
from pras.repair import repair_heavy_atoms
from pras.repair.heavy_atoms import strip_hydrogens

HEAVY_CASE_IDS: tuple[str, ...] = (
    "1aho-heavy-default",
    "1cjc-heavy-keep-ligand",
    "1aar-heavy-chain-1",
)


def test_repair_heavy_atoms_matches_representative_regressions() -> None:
    """New heavy-atom repair should preserve representative semantics."""

    for case_id in HEAVY_CASE_IDS:
        expected = REPRESENTATIVE_CASES[case_id]
        options = options_for_case(case_id)
        structure = read_structure(expected.input_path, options=options)

        result = repair_heavy_atoms(structure)
        summary = summarize_structure(result.structure)

        assert summary == expected.summary
        assert not result.has_errors()


def test_strip_hydrogens_returns_original_structure_when_no_hydrogens() -> None:
    """Heavy-only structures should bypass no-op hydrogen stripping."""

    structure = ProteinStructure(
        chains=(
            read_structure(Path("tests/fixtures/corpus/pdb2xbi.ent")).chains[0],
        ),
        ligands=(),
        source_format=FileFormat.PDB,
    )

    stripped = strip_hydrogens(structure)

    assert stripped is structure


def options_for_case(case_id: str) -> ProcessOptions:
    """Return ingress options matching a representative heavy scenario."""

    if case_id == "1cjc-heavy-keep-ligand":
        return ProcessOptions(ligand_policy=LigandPolicy.KEEP)

    if case_id == "1aar-heavy-chain-1":
        return ProcessOptions(selected_chain_ids=("A",))

    return ProcessOptions()
