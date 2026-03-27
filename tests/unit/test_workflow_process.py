"""Workflow-spine tests over path and canonical structure sources."""


import pytest
from tests.support.representative_cases import REPRESENTATIVE_CASES
from tests.support.structure_summary import summarize_structure

from pras.api import process_structure
from pras.model import (
    AnalysisKind,
    Atom,
    Chain,
    FileFormat,
    HydrogenPolicy,
    LigandPolicy,
    ProteinStructure,
    Residue,
    ResidueId,
    Vec3,
)
from pras.packing import PackingMode, PackingResult, PackingScope, PackingSpec
from pras.process import ProcessOptions
from pras.workflow.process import apply_canonical_boundary_options

WORKFLOW_CASE_IDS: tuple[str, ...] = (
    "1aho-hydrogen-default",
    "1cjc-heavy-keep-ligand",
    "1cjc-hydrogen-keep-ligand",
    "1afc-hydrogen-his-protonated",
)


def test_process_structure_matches_representative_regressions() -> None:
    """Workflow source dispatch should preserve representative semantics."""

    for case_id in WORKFLOW_CASE_IDS:
        expected = REPRESENTATIVE_CASES[case_id]
        result = process_structure(
            expected.input_path,
            options=options_for_case(case_id),
        )
        summary = summarize_structure(result.structure)

        assert summary == expected.summary
        assert not result.has_errors()


def test_process_structure_applies_boundary_options_to_canonical_input() -> None:
    """Canonical inputs should still honor chain and ligand boundary options."""

    structure = ProteinStructure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),
                ),
            ),
            build_chain(
                "B",
                (
                    build_residue("GLY", "B", 2, ("N", "CA", "C", "O")),
                ),
            ),
        ),
        ligands=(
            build_residue("FAD", "B", 99, ("C1", "N1", "O1"), is_hetero=True),
        ),
        source_format=FileFormat.PDB,
        source_name="canonical-fixture",
    )

    result = process_structure(
        structure,
        options=ProcessOptions(
            selected_chain_ids=("B",),
            ligand_policy=LigandPolicy.KEEP,
        ),
    )

    assert result.structure.chain_ids() == ("B",)
    assert tuple(ligand.component_id for ligand in result.structure.ligands) == (
        "FAD",
    )


def test_process_structure_can_drop_ligands_for_canonical_input() -> None:
    """Canonical inputs should still honor ligand dropping."""

    structure = ProteinStructure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),
                ),
            ),
        ),
        ligands=(
            build_residue("FAD", "A", 99, ("C1", "N1", "O1"), is_hetero=True),
        ),
        source_format=FileFormat.PDB,
        source_name="canonical-fixture",
    )

    result = process_structure(
        structure,
        options=ProcessOptions(ligand_policy=LigandPolicy.DROP),
    )

    assert result.structure.ligands == ()


def test_apply_canonical_boundary_options_returns_original_structure_on_noop() -> None:
    """Canonical boundary normalization should avoid no-op structure copies."""

    structure = ProteinStructure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="canonical-fixture",
    )

    normalized = apply_canonical_boundary_options(
        structure,
        options=ProcessOptions(),
    )

    assert normalized is structure


def test_process_structure_rejects_unimplemented_analysis_requests() -> None:
    """Analysis requests should fail explicitly until analysis is implemented."""

    structure = ProteinStructure(chains=(), ligands=(), source_format=FileFormat.PDB)

    with pytest.raises(NotImplementedError):
        process_structure(
            structure,
            options=ProcessOptions(
                analyses=frozenset({AnalysisKind.RAMACHANDRAN})
            ),
        )


def test_process_structure_uses_packed_reference_for_missing_heavy_atoms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Packing backends should guide missing heavy-atom repair."""

    structure = ProteinStructure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="packing-guidance",
    )

    packed_structure = ProteinStructure(
        chains=(
            Chain(
                chain_id="A",
                residues=(
                    build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")).with_atom(
                        Atom(
                            name="CB",
                            element="C",
                            position=Vec3(99.0, 98.0, 97.0),
                            occupancy=1.0,
                            b_factor=20.0,
                        )
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="packing-guidance",
    )

    expected_residue_id = ResidueId(chain_id="A", seq_num=1)

    class FakePackingBackend:
        def pack(self, request):
            return PackingResult(
                packed_structure=packed_structure,
                changed_residue_ids=(expected_residue_id,),
                issues=(),
                backend_name="faspr",
            )

    def resolve_backend(backend_name: str):
        assert backend_name == "faspr"
        return FakePackingBackend()

    monkeypatch.setattr(
        "pras.workflow.process.resolve_sidechain_packing_backend",
        resolve_backend,
    )

    result = process_structure(
        structure,
        options=ProcessOptions(
            sidechain_packing=PackingSpec(
                backend_name="faspr",
                mode=PackingMode.PACK,
                scope=PackingScope.FULL,
            )
        ),
    )

    residue = result.structure.chain("A").residues[0]

    assert residue.has_atom("CB")
    assert residue.atom("CB").position == Vec3(99.0, 98.0, 97.0)
    assert not result.has_errors()


def options_for_case(case_id: str) -> ProcessOptions:
    """Return workflow options matching one representative scenario."""

    if case_id == "1cjc-heavy-keep-ligand":
        return ProcessOptions(ligand_policy=LigandPolicy.KEEP)

    if case_id == "1aho-hydrogen-default":
        return ProcessOptions(hydrogen_policy=HydrogenPolicy.ADD_MISSING)

    if case_id == "1cjc-hydrogen-keep-ligand":
        return ProcessOptions(
            ligand_policy=LigandPolicy.KEEP,
            hydrogen_policy=HydrogenPolicy.ADD_MISSING,
        )

    if case_id == "1afc-hydrogen-his-protonated":
        return ProcessOptions(
            hydrogen_policy=HydrogenPolicy.ADD_MISSING,
            protonate_histidines=True,
        )

    return ProcessOptions()


def build_chain(chain_id: str, residues: tuple[Residue, ...]):
    """Build a canonical chain for workflow tests."""

    return Chain(chain_id=chain_id, residues=residues)


def build_residue(
    component_id: str,
    chain_id: str,
    seq_num: int,
    atom_names: tuple[str, ...],
    *,
    is_hetero: bool = False,
) -> Residue:
    """Build a canonical residue for workflow tests."""

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

    preset_positions = (
        Vec3(0.000, 0.000, 0.000),
        Vec3(1.458, 0.000, 0.000),
        Vec3(2.028, 1.417, 0.000),
        Vec3(3.235, 1.593, 0.248),
        Vec3(1.145, -0.842, 1.074),
        Vec3(2.318, -1.152, 1.556),
    )
    position = preset_positions[(atom_index - 1) % len(preset_positions)]

    return Atom(
        name=atom_name,
        element=infer_element(atom_name),
        position=position,
        occupancy=1.0,
        b_factor=20.0,
    )


def infer_element(atom_name: str) -> str:
    """Infer a simple test element from an atom name."""

    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        raise ValueError(f"atom_name must contain at least one letter: {atom_name}")

    return letters[0]
