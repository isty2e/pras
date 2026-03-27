"""Unit tests for generic side-chain packing request and result entities."""

import pytest

from pras.errors import PackingError, ResidueNotFoundError
from pras.model import (
    Atom,
    Chain,
    FileFormat,
    IssueSeverity,
    ProteinStructure,
    Residue,
    ResidueId,
    ValidationIssue,
    ValidationIssueKind,
    Vec3,
)
from pras.packing import (
    PackingAlphabet,
    PackingCapabilities,
    PackingMode,
    PackingPlan,
    PackingRequest,
    PackingResult,
    PackingScope,
    PackingSelection,
    PackingSpec,
)


def test_packing_spec_normalizes_backend_name_and_residue_lists() -> None:
    """Packing specs should normalize backend names and deduplicate residue ids."""

    residue_id = ResidueId(chain_id="A", seq_num=10)
    spec = PackingSpec(
        backend_name=" FASPR ",
        mode=PackingMode.PACK,
        scope=PackingScope.LOCAL,
        mutable_residue_ids=(residue_id, residue_id),
    )

    assert spec.backend_name == "faspr"
    assert spec.mutable_residue_ids == (residue_id,)
    assert spec.referenced_residue_ids() == (residue_id,)
    assert spec.is_local()


def test_packing_spec_rejects_invalid_local_or_overlapping_ids() -> None:
    """Packing specs should reject invalid local and overlapping residue sets."""

    residue_id = ResidueId(chain_id="A", seq_num=10)

    with pytest.raises(ValueError, match="local side-chain packing"):
        PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL,
        )

    with pytest.raises(ValueError, match="must not overlap"):
        PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL,
            mutable_residue_ids=(residue_id,),
            frozen_residue_ids=(residue_id,),
        )


def test_packing_capabilities_enforce_supported_surface() -> None:
    """Capability declarations should reject unsupported specs."""

    capabilities = PackingCapabilities(
        supports_full_structure_packing=False,
        supports_local_packing=True,
        supports_partial_sequence=False,
        supports_refinement=False,
        supports_noncanonical_components=False,
        deterministic_given_same_inputs=True,
    )
    supported_spec = PackingSpec(
        backend_name="hpacker",
        scope=PackingScope.LOCAL,
        mutable_residue_ids=(ResidueId(chain_id="A", seq_num=10),),
    )
    unsupported_spec = PackingSpec(
        backend_name="hpacker",
        mode=PackingMode.REFINE,
        scope=PackingScope.LOCAL,
        mutable_residue_ids=(ResidueId(chain_id="A", seq_num=10),),
    )

    assert capabilities.supports_spec(supported_spec)
    assert not capabilities.supports_spec(unsupported_spec)

    with pytest.raises(ValueError, match="refinement mode"):
        capabilities.require_support_for(unsupported_spec)


def test_packing_request_validates_referenced_residues_and_sequence_length() -> None:
    """Packing requests should reject residue ids or sequences that do not match."""

    structure = build_structure()
    mutable_residue_ids = (
        ResidueId(chain_id="A", seq_num=1),
        ResidueId(chain_id="A", seq_num=2),
    )

    request = PackingRequest(
        structure=structure,
        spec=PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL,
            mutable_residue_ids=mutable_residue_ids,
            target_sequence="AG",
        ),
    )

    assert request.referenced_residue_ids() == mutable_residue_ids
    assert request.referenced_residue_count() == 2

    with pytest.raises(ValueError, match="target_sequence length"):
        PackingRequest(
            structure=structure,
            spec=PackingSpec(
                backend_name="faspr",
                scope=PackingScope.LOCAL,
                mutable_residue_ids=mutable_residue_ids,
                target_sequence="A",
            ),
        )

    with pytest.raises(ResidueNotFoundError):
        PackingRequest(
            structure=structure,
            spec=PackingSpec(
                backend_name="faspr",
                scope=PackingScope.LOCAL,
                mutable_residue_ids=(ResidueId(chain_id="A", seq_num=9),),
            ),
        )


def test_packing_plan_resolves_selection_and_effective_sequence() -> None:
    """Packing plans should own selection and mutation semantics."""

    structure = build_structure()
    request = PackingRequest(
        structure=structure,
        spec=PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL,
            mutable_residue_ids=(ResidueId(chain_id="A", seq_num=2),),
            frozen_residue_ids=(ResidueId(chain_id="A", seq_num=1),),
            target_sequence="V",
        ),
    )
    plan = PackingPlan.from_request(request)
    alphabet = PackingAlphabet({"GLY": "G", "ALA": "A"})

    assert plan.selected_residue_ids() == (ResidueId(chain_id="A", seq_num=2),)
    assert plan.fixed_residue_ids() == (
        ResidueId(chain_id="A", seq_num=1),
    )
    assert plan.selected_residue_count() == 1
    assert plan.original_sequence_tokens(alphabet) == ("G", "A")
    assert plan.effective_sequence_tokens(alphabet) == ("G", "V")


def test_packing_plan_detects_changed_residues_in_packed_output() -> None:
    """Packing plans should compare packed structures at the domain layer."""

    structure = build_structure()
    request = PackingRequest(
        structure=structure,
        spec=PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )
    plan = PackingPlan.from_request(request)
    changed_structure = structure.with_updated_chain(
        Chain(
            chain_id="A",
            residues=(
                structure.chain("A").residues[0],
                structure.chain("A").residues[1].with_atom(
                    Atom(
                        name="CB",
                        element="C",
                        position=Vec3(99.0, 98.0, 97.0),
                        occupancy=1.0,
                        b_factor=20.0,
                    )
                ),
            ),
        )
    )

    assert plan.changed_residue_ids_after(changed_structure) == (
        ResidueId(chain_id="A", seq_num=2),
    )


def test_packing_selection_rejects_unknown_polymer_residues() -> None:
    """Packing selections should only reference residues present in the plan."""

    with pytest.raises(ValueError, match="must belong to polymer residues"):
        PackingSelection(
            scope=PackingScope.LOCAL,
            polymer_residue_ids=(ResidueId(chain_id="A", seq_num=1),),
            mutable_residue_ids=(ResidueId(chain_id="A", seq_num=2),),
        )


def test_packing_plan_rejects_full_sequence_length_mismatch() -> None:
    """Full-sequence overrides should match the polymer residue count."""

    structure = build_structure()
    plan = PackingPlan.from_request(
        PackingRequest(
            structure=structure,
            spec=PackingSpec(
                backend_name="faspr",
                scope=PackingScope.FULL,
                target_sequence="A",
            ),
        )
    )

    with pytest.raises(PackingError, match="polymer residue count"):
        plan.effective_sequence_tokens(PackingAlphabet({"GLY": "G", "ALA": "A"}))


def test_packing_result_normalizes_backend_metadata_and_validates_changes() -> None:
    """Packing results should normalize backend metadata and changed residues."""

    structure = build_structure()
    residue_id = ResidueId(chain_id="A", seq_num=1)
    result = PackingResult(
        packed_structure=structure,
        changed_residue_ids=(residue_id, residue_id),
        issues=(
            ValidationIssue(
                kind=ValidationIssueKind.UNSUPPORTED_COMPONENT,
                severity=IssueSeverity.WARNING,
                message="backend skipped one residue",
                residue_id=residue_id,
            ),
        ),
        backend_name=" FASPR ",
        backend_version=" 1.0.0 ",
    )

    assert result.backend_name == "faspr"
    assert result.backend_version == "1.0.0"
    assert result.changed_residue_ids == (residue_id,)
    assert result.changed_residue(residue_id)
    assert result.changed_residue_count() == 1
    assert result.has_issues()


def build_structure() -> ProteinStructure:
    """Build a minimal canonical structure for packing tests."""

    return ProteinStructure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),
                    build_residue("ALA", "A", 2, ("N", "CA", "C", "O", "CB")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="packing-fixture",
    )


def build_chain(chain_id: str, residues: tuple[Residue, ...]):
    """Build one canonical chain for a unit fixture."""

    return Chain(chain_id=chain_id, residues=residues)


def build_residue(
    component_id: str,
    chain_id: str,
    seq_num: int,
    atom_names: tuple[str, ...],
) -> Residue:
    """Build a canonical residue for a unit fixture."""

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
    """Build one deterministic canonical atom for a unit fixture."""

    position = Vec3(float(atom_index), float(atom_index + 1), float(atom_index + 2))
    return Atom(
        name=atom_name,
        element=infer_element(atom_name),
        position=position,
        occupancy=1.0,
        b_factor=20.0,
    )


def infer_element(atom_name: str) -> str:
    """Infer a simple element token from an atom name."""

    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        raise ValueError(f"atom_name must contain at least one letter: {atom_name}")

    return letters[0]
