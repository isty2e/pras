"""Adversarial edge cases for component-template semantics."""

from dataclasses import replace

import pytest

from pras.chemistry import (
    ChemicalComponentDefinition,
    ComponentLibrary,
    ForceFieldAtomParams,
    ResidueTemplate,
    build_standard_component_library,
)
from pras.chemistry.component import (
    HeavyAtomBuilderKind,
    HeavyAtomSemantics,
    HydrogenSemantics,
    RotatableHydrogenKind,
)
from pras.model import (
    Atom,
    Chain,
    FileFormat,
    ProteinStructure,
    RepairEventKind,
    Residue,
    ResidueId,
    Vec3,
)
from pras.repair import add_hydrogens, repair_heavy_atoms
from pras.repair.hydrogen_engine import (
    evaluate_operation,
    resolved_coordinate,
    resolved_float,
)


def make_template(
    component_id: str,
    *,
    aliases: tuple[str, ...] = (),
) -> ResidueTemplate:
    """Build a minimal residue template for edge-case tests."""

    return ResidueTemplate(
        definition=ChemicalComponentDefinition(
            component_id=component_id,
            atom_names=("N", "CA", "C", "O"),
            aliases=aliases,
        )
    )


def test_component_library_rejects_alias_collision_between_templates() -> None:
    """Two templates must not silently share the same alias."""

    with pytest.raises(ValueError, match="ambiguous alias"):
        ComponentLibrary(
            templates={
                "HIS": make_template("HIS", aliases=("HSE",)),
                "MSE": make_template("MSE", aliases=("HSE",)),
            }
        )


def test_component_library_rejects_alias_collision_with_component_id() -> None:
    """A template alias must not shadow another template's canonical id."""

    with pytest.raises(ValueError, match="ambiguous alias"):
        ComponentLibrary(
            templates={
                "HIS": make_template("HIS", aliases=("MSE",)),
                "MSE": make_template("MSE"),
            }
        )


def test_heavy_atom_semantics_rejects_duplicate_atom_names() -> None:
    """Heavy-atom semantics should reject duplicate atom-order entries."""

    with pytest.raises(ValueError, match="unique"):
        HeavyAtomSemantics(
            builder_kind=HeavyAtomBuilderKind.ALA,
            atom_order=("N", "CA", "CA"),
        )


def test_hydrogen_semantics_rejects_rotatable_and_static_configuration_mix() -> None:
    """Hydrogen semantics must not mix rotatable and static planning modes."""

    with pytest.raises(ValueError, match="either"):
        HydrogenSemantics(
            plan_with_backbone=((("HA",), "class3", ("CB", "N", "CA")),),
            rotatable_kind=RotatableHydrogenKind.SER,
        )


def test_hydrogen_semantics_rejects_noop_configuration() -> None:
    """Hydrogen semantics should reject configurations with no executable plan."""

    with pytest.raises(ValueError, match="require"):
        HydrogenSemantics()


def test_hydrogen_semantics_rejects_without_backbone_only_plan() -> None:
    """Backbone-conditional plans require a primary with-backbone plan."""

    with pytest.raises(ValueError, match="with_backbone"):
        HydrogenSemantics(
            plan_without_backbone=((("HA",), "class3", ("CB", "N", "CA")),),
        )


def test_repair_heavy_atoms_preserves_ligand_hydrogens() -> None:
    """Direct heavy-atom repair should not strip ligand hydrogens."""

    polymer_residue = Residue(
        component_id="GLY",
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atoms=(
            Atom("N", "N", Vec3(1.0, 1.0, 1.0)),
            Atom("CA", "C", Vec3(2.0, 1.5, 1.0)),
            Atom("C", "C", Vec3(3.0, 1.0, 1.5)),
            Atom("O", "O", Vec3(3.8, 1.2, 2.3)),
        ),
    )
    ligand = Residue(
        component_id="LIG",
        residue_id=ResidueId(chain_id="A", seq_num=2),
        atoms=(
            Atom("C1", "C", Vec3(7.0, 7.0, 7.0)),
            Atom("H1", "H", Vec3(7.8, 7.0, 7.0)),
        ),
        is_hetero=True,
    )
    structure = ProteinStructure(
        chains=(Chain(chain_id="A", residues=(polymer_residue,)),),
        ligands=(ligand,),
        source_format=FileFormat.PDB,
        source_name="ligand-hydrogen-edge",
    )

    result = repair_heavy_atoms(structure)

    assert result.structure.ligands[0].atom_names() == ("C1", "H1")


def test_custom_alias_template_normalizes_and_repairs_heavy_atoms() -> None:
    """A custom alias should resolve to the canonical template during repair."""

    library = build_standard_component_library()
    ala_template = library.require("ALA")
    aliased_template = replace(
        ala_template,
        definition=replace(ala_template.definition, aliases=("DAL",)),
    )
    custom_library = library.with_template(aliased_template)
    structure = ProteinStructure(
        chains=(
            Chain(
                chain_id="A",
                residues=(
                    Residue(
                        component_id="DAL",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            Atom("N", "N", Vec3(1.0, 1.0, 1.0)),
                            Atom("CA", "C", Vec3(2.0, 1.5, 1.0)),
                            Atom("C", "C", Vec3(3.0, 1.0, 1.5)),
                            Atom("O", "O", Vec3(3.8, 1.2, 2.3)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="alias-heavy-repair",
    )

    result = repair_heavy_atoms(structure, component_library=custom_library)
    residue = result.structure.chain("A").residues[0]

    assert residue.component_id == "ALA"
    assert residue.has_atom("CB")
    assert any(
        event.kind is RepairEventKind.COMPONENT_NORMALIZED for event in result.repairs
    )
    assert any(
        event.kind is RepairEventKind.HEAVY_ATOMS_ADDED for event in result.repairs
    )


def test_integer_hydrogen_plan_arguments_are_accepted_end_to_end() -> None:
    """Integer-valued static hydrogen plans should be executable."""

    library = build_standard_component_library()
    ala_template = library.require("ALA")
    custom_template = replace(
        ala_template,
        hydrogen_semantics=HydrogenSemantics(
            plan_with_backbone=(
                (("HX",), "calcCoordinate", ("C", "CA", "CB", 1, 180, 109)),
            ),
        ),
    )
    custom_library = library.with_template(custom_template)
    structure = ProteinStructure(
        chains=(
            Chain(
                chain_id="A",
                residues=(
                    Residue(
                        component_id="ALA",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            Atom("N", "N", Vec3(1.0, 1.0, 1.0)),
                            Atom("CA", "C", Vec3(2.0, 1.5, 1.0)),
                            Atom("C", "C", Vec3(3.0, 1.0, 1.5)),
                            Atom("O", "O", Vec3(3.8, 1.2, 2.3)),
                            Atom("CB", "C", Vec3(2.0, 2.6, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="integer-hydrogen-plan",
    )

    result = add_hydrogens(structure, component_library=custom_library)

    assert result.structure.chain("A").residues[0].has_atom("HX")


def test_rotatable_hydrogen_tolerates_missing_neighbor_forcefield_params() -> None:
    """Rotatable-hydrogen refinement should tolerate sparse neighbor parameters."""

    library = build_standard_component_library()
    gly_template = library.require("GLY")
    custom_library = library.with_template(
        replace(gly_template, forcefield_parameters={})
    )
    structure = ProteinStructure(
        chains=(
            Chain(
                chain_id="A",
                residues=(
                    Residue(
                        component_id="SER",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            Atom("N", "N", Vec3(1.0, 1.0, 1.0)),
                            Atom("CA", "C", Vec3(2.0, 1.5, 1.0)),
                            Atom("C", "C", Vec3(3.0, 1.0, 1.5)),
                            Atom("O", "O", Vec3(3.8, 1.2, 2.3)),
                            Atom("CB", "C", Vec3(2.0, 2.6, 0.0)),
                            Atom("OG", "O", Vec3(2.8, 3.5, -0.5)),
                        ),
                    ),
                    Residue(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=2),
                        atoms=(
                            Atom("N", "N", Vec3(4.0, 0.8, 1.7)),
                            Atom("CA", "C", Vec3(5.0, 1.2, 2.1)),
                            Atom("C", "C", Vec3(5.9, 0.4, 3.0)),
                            Atom("O", "O", Vec3(6.9, 0.8, 3.4)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="missing-forcefield-neighbor",
    )

    result = add_hydrogens(structure, component_library=custom_library)

    assert result.structure.chain("A").residues[0].has_atom("HG")


def test_explicit_alias_map_resolves_lowercase_alias_during_heavy_repair() -> None:
    """Explicit alias maps should normalize lowercase aliases into templates."""

    template = replace(
        build_standard_component_library().require("GLY"),
        definition=replace(
            build_standard_component_library().require("GLY").definition,
            aliases=(),
        ),
    )
    library = ComponentLibrary(
        templates={"GLY": template},
        alias_to_component_id={"dgly": "gly"},
    )
    structure = ProteinStructure(
        chains=(
            Chain(
                chain_id="A",
                residues=(
                    Residue(
                        component_id="DGLY",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            Atom("N", "N", Vec3(1.0, 1.0, 1.0)),
                            Atom("CA", "C", Vec3(2.0, 1.5, 1.0)),
                            Atom("C", "C", Vec3(3.0, 1.0, 1.5)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="explicit-alias-map",
    )

    result = repair_heavy_atoms(structure, component_library=library)
    residue = result.structure.chain("A").residues[0]

    assert residue.component_id == "GLY"
    assert residue.has_atom("O")


def test_lowercase_forcefield_parameter_keys_are_normalized() -> None:
    """Template force-field parameter keys should normalize to canonical atom names."""

    template = ResidueTemplate(
        definition=ChemicalComponentDefinition(
            component_id="SER",
            atom_names=("N", "CA", "C", "O", "CB", "OG"),
        ),
        forcefield_parameters={
            "og": ForceFieldAtomParams(0.1, 0.2, 0.3),
            "cb": ForceFieldAtomParams(0.1, 0.2, 0.3),
        },
    )

    assert template.has_forcefield_params("OG")
    assert template.has_forcefield_params("CB")


def test_repair_heavy_atoms_ligand_only_input_preserves_ligand_hydrogens() -> None:
    """Ligand-only heavy repair should preserve ligand hydrogens unchanged."""

    ligand = Residue(
        component_id="LIG",
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atoms=(
            Atom("C1", "C", Vec3(7.0, 7.0, 7.0)),
            Atom("H1", "H", Vec3(7.8, 7.0, 7.0)),
        ),
        is_hetero=True,
    )
    structure = ProteinStructure(
        chains=(),
        ligands=(ligand,),
        source_format=FileFormat.PDB,
        source_name="ligand-only-heavy-repair",
    )

    result = repair_heavy_atoms(structure)

    assert result.structure.chains == ()
    assert result.structure.ligands[0].atom_names() == ("C1", "H1")


def test_repair_heavy_atoms_strips_polymer_hydrogens_but_keeps_ligands() -> None:
    """Direct heavy repair should strip polymer hydrogens without touching ligands."""

    polymer_residue = Residue(
        component_id="GLY",
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atoms=(
            Atom("N", "N", Vec3(1.0, 1.0, 1.0)),
            Atom("CA", "C", Vec3(2.0, 1.5, 1.0)),
            Atom("C", "C", Vec3(3.0, 1.0, 1.5)),
            Atom("O", "O", Vec3(3.8, 1.2, 2.3)),
            Atom("H1", "H", Vec3(0.5, 1.3, 1.2)),
        ),
    )
    ligand = Residue(
        component_id="LIG",
        residue_id=ResidueId(chain_id="A", seq_num=2),
        atoms=(
            Atom("C1", "C", Vec3(7.0, 7.0, 7.0)),
            Atom("H1", "H", Vec3(7.8, 7.0, 7.0)),
        ),
        is_hetero=True,
    )
    structure = ProteinStructure(
        chains=(Chain(chain_id="A", residues=(polymer_residue,)),),
        ligands=(ligand,),
        source_format=FileFormat.PDB,
        source_name="polymer-vs-ligand-hydrogens",
    )

    result = repair_heavy_atoms(structure)
    repaired_residue = result.structure.chain("A").residues[0]

    assert "H1" not in repaired_residue.atom_names()
    assert result.structure.ligands[0].atom_names() == ("C1", "H1")


def test_repair_heavy_atoms_does_not_duplicate_existing_oxt() -> None:
    """Terminal OXT repair should not duplicate an already-present atom."""

    residue = Residue(
        component_id="GLY",
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atoms=(
            Atom("N", "N", Vec3(1.0, 1.0, 1.0)),
            Atom("CA", "C", Vec3(2.0, 1.5, 1.0)),
            Atom("C", "C", Vec3(3.0, 1.0, 1.5)),
            Atom("O", "O", Vec3(3.8, 1.2, 2.3)),
            Atom("OXT", "O", Vec3(3.6, 0.2, 1.0)),
        ),
    )
    structure = ProteinStructure(
        chains=(Chain(chain_id="A", residues=(residue,)),),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="existing-oxt",
    )

    result = repair_heavy_atoms(structure)
    repaired = result.structure.chain("A").residues[0]

    assert repaired.atom_names().count("OXT") == 1
    assert not any(
        event.kind is RepairEventKind.C_TERMINAL_OXT_ADDED for event in result.repairs
    )


def test_add_hydrogens_with_custom_alias_library_preserves_ligand_hydrogens() -> None:
    """Hydrogenation should preserve ligand hydrogens while using custom aliases."""

    library = build_standard_component_library()
    gly_template = library.require("GLY")
    aliased_template = replace(
        gly_template,
        definition=replace(gly_template.definition, aliases=("DGLY",)),
    )
    custom_library = library.with_template(aliased_template)
    polymer_residue = Residue(
        component_id="DGLY",
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atoms=(
            Atom("N", "N", Vec3(1.0, 1.0, 1.0)),
            Atom("CA", "C", Vec3(2.0, 1.5, 1.0)),
            Atom("C", "C", Vec3(3.0, 1.0, 1.5)),
            Atom("O", "O", Vec3(3.8, 1.2, 2.3)),
        ),
    )
    ligand = Residue(
        component_id="LIG",
        residue_id=ResidueId(chain_id="A", seq_num=2),
        atoms=(
            Atom("C1", "C", Vec3(7.0, 7.0, 7.0)),
            Atom("H1", "H", Vec3(7.8, 7.0, 7.0)),
        ),
        is_hetero=True,
    )
    structure = ProteinStructure(
        chains=(Chain(chain_id="A", residues=(polymer_residue,)),),
        ligands=(ligand,),
        source_format=FileFormat.PDB,
        source_name="custom-alias-hydrogenation",
    )

    result = add_hydrogens(structure, component_library=custom_library)

    assert result.structure.chain("A").residues[0].component_id == "GLY"
    assert result.structure.ligands[0].atom_names() == ("C1", "H1")


def test_rotatable_hydrogen_tolerates_missing_current_forcefield_params() -> None:
    """Rotatable hydrogen placement should tolerate missing current-residue params."""

    library = build_standard_component_library()
    ser_template = library.require("SER")
    custom_library = library.with_template(
        replace(ser_template, forcefield_parameters={})
    )
    structure = ProteinStructure(
        chains=(
            Chain(
                chain_id="A",
                residues=(
                    Residue(
                        component_id="SER",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            Atom("N", "N", Vec3(1.0, 1.0, 1.0)),
                            Atom("CA", "C", Vec3(2.0, 1.5, 1.0)),
                            Atom("C", "C", Vec3(3.0, 1.0, 1.5)),
                            Atom("O", "O", Vec3(3.8, 1.2, 2.3)),
                            Atom("CB", "C", Vec3(2.0, 2.6, 0.0)),
                            Atom("OG", "O", Vec3(2.8, 3.5, -0.5)),
                        ),
                    ),
                    Residue(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=2),
                        atoms=(
                            Atom("N", "N", Vec3(4.0, 0.8, 1.7)),
                            Atom("CA", "C", Vec3(5.0, 1.2, 2.1)),
                            Atom("C", "C", Vec3(5.9, 0.4, 3.0)),
                            Atom("O", "O", Vec3(6.9, 0.8, 3.4)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="missing-current-forcefield",
    )

    result = add_hydrogens(structure, component_library=custom_library)

    assert result.structure.chain("A").residues[0].has_atom("HG")


def test_component_library_with_template_preserves_existing_aliases() -> None:
    """Adding one template should not break previously registered aliases."""

    library = build_standard_component_library()
    gly_template = library.require("GLY")
    extended_library = library.with_template(
        replace(
            gly_template,
            definition=replace(gly_template.definition, aliases=("DGLY",)),
        )
    )

    assert extended_library.normalize_component_id("HSE") == "HIS"
    assert extended_library.normalize_component_id("dgly") == "GLY"


def test_hydrogen_semantics_static_plan_prefers_without_backbone_variant() -> None:
    """Backbone-conditional static plans should select the alternate variant."""

    semantics = HydrogenSemantics(
        plan_with_backbone=((("HB",), "class3", ("CB", "N", "CA")),),
        plan_without_backbone=((("HX",), "class3", ("CB", "N", "CA")),),
    )

    assert semantics.static_plan(include_backbone_hydrogen=True) == (
        (("HB",), "class3", ("CB", "N", "CA")),
    )
    assert semantics.static_plan(include_backbone_hydrogen=False) == (
        (("HX",), "class3", ("CB", "N", "CA")),
    )


def test_hydrogen_semantics_static_plan_falls_back_to_primary_plan() -> None:
    """Static plans without an alternate should reuse the primary variant."""

    semantics = HydrogenSemantics(
        plan_with_backbone=((("HB",), "class3", ("CB", "N", "CA")),),
    )

    assert semantics.static_plan(include_backbone_hydrogen=False) == (
        (("HB",), "class3", ("CB", "N", "CA")),
    )


def test_residue_template_missing_atom_names_normalizes_inputs() -> None:
    """Missing-atom checks should normalize present and excluded atom names."""

    template = ResidueTemplate(
        definition=ChemicalComponentDefinition(
            component_id="SER",
            atom_names=("N", "CA", "C", "O", "CB", "OG", "OXT"),
        )
    )

    assert template.missing_atom_names(
        ("n", "ca", "c", "o"),
        exclude_atom_names=("oxt",),
    ) == ("CB", "OG")


def test_component_library_with_template_rejects_new_alias_collision() -> None:
    """Adding a template should still reject alias conflicts against the library."""

    library = ComponentLibrary(
        templates={
            "HIS": make_template("HIS", aliases=("HSE",)),
            "GLY": make_template("GLY"),
        }
    )

    with pytest.raises(ValueError, match="ambiguous alias"):
        library.with_template(make_template("MSE", aliases=("HSE",)))


def test_hydrogen_operation_rejects_unknown_method_name() -> None:
    """Hydrogen geometry DSL should reject unknown operation names."""

    with pytest.raises(ValueError, match="unsupported hydrogen geometry method"):
        evaluate_operation(
            "class999",
            ("CB", "N", "CA"),
            atom_coordinates={
                "CB": [1.0, 0.0, 0.0],
                "N": [0.0, 0.0, 0.0],
                "CA": [0.0, 1.0, 0.0],
            },
        )


def test_hydrogen_operation_argument_type_guards_hold() -> None:
    """Hydrogen geometry DSL should reject swapped numeric and atom arguments."""

    with pytest.raises(TypeError, match="coordinate arguments"):
        resolved_coordinate(1.0, {"CA": [0.0, 0.0, 0.0]})

    with pytest.raises(TypeError, match="numeric hydrogen-plan arguments"):
        resolved_float("CA")


def test_custom_alias_heavy_repair_emits_single_normalization_event() -> None:
    """Alias-driven heavy repair should normalize exactly once per residue."""

    library = build_standard_component_library()
    ser_template = library.require("SER")
    custom_library = library.with_template(
        replace(
            ser_template,
            definition=replace(ser_template.definition, aliases=("DSER",)),
        )
    )
    structure = ProteinStructure(
        chains=(
            Chain(
                chain_id="A",
                residues=(
                    Residue(
                        component_id="DSER",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            Atom("N", "N", Vec3(1.0, 1.0, 1.0)),
                            Atom("CA", "C", Vec3(2.0, 1.5, 1.0)),
                            Atom("C", "C", Vec3(3.0, 1.0, 1.5)),
                            Atom("O", "O", Vec3(3.8, 1.2, 2.3)),
                            Atom("CB", "C", Vec3(2.0, 2.6, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="single-normalization-event",
    )

    result = repair_heavy_atoms(structure, component_library=custom_library)
    normalization_events = [
        event
        for event in result.repairs
        if event.kind is RepairEventKind.COMPONENT_NORMALIZED
    ]

    assert len(normalization_events) == 1
    assert normalization_events[0].component_id == "SER"


def test_integer_plan_single_residue_does_not_add_propagated_backbone_hydrogen(
) -> None:
    """Single-residue custom hydrogen plans should not manufacture propagated H."""

    library = build_standard_component_library()
    ala_template = library.require("ALA")
    custom_library = library.with_template(
        replace(
            ala_template,
            hydrogen_semantics=HydrogenSemantics(
                plan_with_backbone=(
                    (("HX",), "calcCoordinate", ("C", "CA", "CB", 1, 180, 109)),
                ),
            ),
        )
    )
    structure = ProteinStructure(
        chains=(
            Chain(
                chain_id="A",
                residues=(
                    Residue(
                        component_id="ALA",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            Atom("N", "N", Vec3(1.0, 1.0, 1.0)),
                            Atom("CA", "C", Vec3(2.0, 1.5, 1.0)),
                            Atom("C", "C", Vec3(3.0, 1.0, 1.5)),
                            Atom("O", "O", Vec3(3.8, 1.2, 2.3)),
                            Atom("CB", "C", Vec3(2.0, 2.6, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="single-residue-custom-plan",
    )

    residue = (
        add_hydrogens(structure, component_library=custom_library)
        .structure.chain("A")
        .residues[0]
    )

    assert residue.has_atom("HX")
    assert "H" not in residue.atom_names()


def test_component_library_accepts_matching_explicit_alias_mapping() -> None:
    """Explicit alias mappings may repeat template aliases when they agree."""

    template = make_template("GLY", aliases=("DGLY",))
    library = ComponentLibrary(
        templates={"GLY": template},
        alias_to_component_id={"dgly": "GLY"},
    )

    assert library.normalize_component_id("DGLY") == "GLY"


def test_class2_operation_accepts_integer_bond_length_argument() -> None:
    """Tetrahedral-pair DSL operations should accept integer bond lengths."""

    coordinates = evaluate_operation(
        "class2",
        ("CA", "CG", "CB", 1),
        atom_coordinates={
            "CA": [1.0, 0.0, 0.0],
            "CG": [0.0, 1.0, 0.0],
            "CB": [0.0, 0.0, 0.0],
        },
    )

    assert len(coordinates) == 2


def test_class5_operation_accepts_integer_bond_length_argument() -> None:
    """Planar-single DSL operations should accept integer bond lengths."""

    coordinates = evaluate_operation(
        "class5",
        ("CE1", "CD1", "CG", 1),
        atom_coordinates={
            "CE1": [1.0, 0.0, 0.0],
            "CD1": [0.0, 0.0, 0.0],
            "CG": [0.0, 1.0, 0.0],
        },
    )

    assert len(coordinates) == 1


def test_class3_operation_returns_one_coordinate() -> None:
    """Single-coordinate tetrahedral DSL operations should stay shape-stable."""

    coordinates = evaluate_operation(
        "class3",
        ("CB", "N", "CA"),
        atom_coordinates={
            "CB": [1.0, 0.0, 0.0],
            "N": [0.0, 0.0, 0.0],
            "CA": [0.0, 1.0, 0.0],
        },
    )

    assert len(coordinates) == 1
