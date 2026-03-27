"""Representative regression tests for hydrogen placement."""

from pathlib import Path

import pytest
from tests.support.representative_cases import REPRESENTATIVE_CASES
from tests.support.structure_summary import summarize_structure

from pras.io import read_structure
from pras.model import LigandPolicy
from pras.process import ProcessOptions
from pras.repair import add_hydrogens

HYDROGEN_CASE_IDS: tuple[str, ...] = (
    "1aho-hydrogen-default",
    "1cjc-hydrogen-keep-ligand",
    "1afc-hydrogen-his-protonated",
)


def test_add_hydrogens_matches_representative_regressions() -> None:
    """New hydrogen placement should preserve representative semantics."""

    for case_id in HYDROGEN_CASE_IDS:
        expected = REPRESENTATIVE_CASES[case_id]
        structure = read_structure(
            expected.input_path,
            options=options_for_case(case_id),
        )

        result = add_hydrogens(
            structure,
            protonate_histidines=expected.protonate_histidines,
        )
        summary = summarize_structure(result.structure)

        assert summary == expected.summary
        assert not result.has_errors()


@pytest.mark.parametrize(
    ("input_path", "expected_digest"),
    (
        pytest.param(
            Path("tests/fixtures/corpus/pdb2a1d.ent"),
            "50e9b4880ac62f9d3e51f4f1886d9cfa70db116a3a445db8b29d8860b8c8c70f",
            id="pdb2a1d",
        ),
        pytest.param(
            Path("tests/fixtures/corpus/pdb2xbi.ent"),
            "bd1b10d9c9facfb4655a5c36d6a17cd4180eb63a5a921f0c4796abf7561f6f85",
            id="pdb2xbi",
        ),
    ),
)
def test_add_hydrogens_matches_known_upstream_hydrogen_digests(
    input_path: Path, expected_digest: str
) -> None:
    """Supported repair fixtures should retain their verified upstream digests."""

    structure = read_structure(input_path, options=ProcessOptions())
    result = add_hydrogens(structure)

    assert summarize_structure(result.structure).semantic_digest == expected_digest


def options_for_case(case_id: str) -> ProcessOptions:
    """Return ingress options matching a representative hydrogen scenario."""

    if case_id == "1cjc-hydrogen-keep-ligand":
        return ProcessOptions(ligand_policy=LigandPolicy.KEEP)

    return ProcessOptions()
