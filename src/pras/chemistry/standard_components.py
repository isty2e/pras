"""Built-in standard-component definitions for baseline protein repair."""

from functools import lru_cache

from pras.chemistry.component import (
    ChemicalComponentDefinition,
    ComponentCapability,
    ForceFieldAtomParams,
    HeavyAtomBuilderKind,
    HeavyAtomSemantics,
    HydrogenSemantics,
    ResidueTemplate,
    RotatableHydrogenKind,
)
from pras.chemistry.component_library import ComponentLibrary
from pras.chemistry.hydrogen_plans import (
    STANDARD_HYDROGEN_PLANS,
    TRP_WITHOUT_BACKBONE_HYDROGEN_PLAN,
)
from pras.chemistry.standard_forcefield import FORCEFIELD_PARAMETERS

STANDARD_COMPONENT_ATOMS: dict[str, tuple[str, ...]] = {
    "ALA": ("N", "CA", "C", "O", "CB", "OXT"),
    "ARG": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "OXT"),
    "ASN": ("N", "CA", "C", "O", "CB", "CG", "ND2", "OD1", "OXT"),
    "ASP": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "OXT"),
    "CYS": ("N", "CA", "C", "O", "CB", "SG", "OXT"),
    "GLN": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE2", "OE1", "OXT"),
    "GLU": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "OXT"),
    "GLY": ("N", "CA", "C", "O", "OXT"),
    "HIS": ("N", "CA", "C", "O", "CB", "CG", "CD2", "ND1", "CE1", "NE2", "OXT"),
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "OXT"),
    "LEU": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "OXT"),
    "LYS": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "OXT"),
    "MET": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE", "OXT"),
    "PHE": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OXT"),
    "PRO": ("N", "CA", "C", "O", "CB", "CG", "CD", "OXT"),
    "SER": ("N", "CA", "C", "O", "CB", "OG", "OXT"),
    "THR": ("N", "CA", "C", "O", "CB", "CG2", "OG1", "OXT"),
    "TRP": (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE2",
        "CE3",
        "NE1",
        "CZ2",
        "CZ3",
        "CH2",
        "OXT",
    ),
    "TYR": (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
        "OXT",
    ),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "OXT"),
}
HISTIDINE_ALIASES: tuple[str, ...] = ("HSD", "HSE", "HIE", "HSP")

STANDARD_HEAVY_ATOM_SEMANTICS: dict[str, HeavyAtomSemantics] = {
    "ALA": HeavyAtomSemantics(HeavyAtomBuilderKind.ALA, ("N", "CA", "C", "O", "CB")),
    "ARG": HeavyAtomSemantics(
        HeavyAtomBuilderKind.ARG,
        ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    ),
    "ASN": HeavyAtomSemantics(
        HeavyAtomBuilderKind.ASN,
        ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
    ),
    "ASP": HeavyAtomSemantics(
        HeavyAtomBuilderKind.ASP,
        ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
    ),
    "CYS": HeavyAtomSemantics(
        HeavyAtomBuilderKind.CYS,
        ("N", "CA", "C", "O", "CB", "SG"),
    ),
    "GLN": HeavyAtomSemantics(
        HeavyAtomBuilderKind.GLN,
        ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
    ),
    "GLU": HeavyAtomSemantics(
        HeavyAtomBuilderKind.GLU,
        ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
    ),
    "GLY": HeavyAtomSemantics(HeavyAtomBuilderKind.GLY, ("N", "CA", "C", "O")),
    "HIS": HeavyAtomSemantics(
        HeavyAtomBuilderKind.HIS,
        ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    ),
    "ILE": HeavyAtomSemantics(
        HeavyAtomBuilderKind.ILE,
        ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
    ),
    "LEU": HeavyAtomSemantics(
        HeavyAtomBuilderKind.LEU,
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
    ),
    "LYS": HeavyAtomSemantics(
        HeavyAtomBuilderKind.LYS,
        ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
    ),
    "MET": HeavyAtomSemantics(
        HeavyAtomBuilderKind.MET,
        ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
    ),
    "PHE": HeavyAtomSemantics(
        HeavyAtomBuilderKind.PHE,
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    ),
    "PRO": HeavyAtomSemantics(
        HeavyAtomBuilderKind.PRO,
        ("N", "CA", "C", "O", "CB", "CG", "CD"),
    ),
    "SER": HeavyAtomSemantics(
        HeavyAtomBuilderKind.SER,
        ("N", "CA", "C", "O", "CB", "OG"),
    ),
    "THR": HeavyAtomSemantics(
        HeavyAtomBuilderKind.THR,
        ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    ),
    "TRP": HeavyAtomSemantics(
        HeavyAtomBuilderKind.TRP,
        (
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "NE1",
            "CE2",
            "CE3",
            "CZ2",
            "CZ3",
            "CH2",
        ),
    ),
    "TYR": HeavyAtomSemantics(
        HeavyAtomBuilderKind.TYR,
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    ),
    "VAL": HeavyAtomSemantics(
        HeavyAtomBuilderKind.VAL,
        ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
    ),
}

STANDARD_HYDROGEN_SEMANTICS: dict[str, HydrogenSemantics] = {
    component_id: HydrogenSemantics(plan_with_backbone=plan)
    for component_id, plan in STANDARD_HYDROGEN_PLANS.items()
}
STANDARD_HYDROGEN_SEMANTICS["CYS"] = HydrogenSemantics(
    rotatable_kind=RotatableHydrogenKind.CYS
)
STANDARD_HYDROGEN_SEMANTICS["SER"] = HydrogenSemantics(
    rotatable_kind=RotatableHydrogenKind.SER
)
STANDARD_HYDROGEN_SEMANTICS["THR"] = HydrogenSemantics(
    rotatable_kind=RotatableHydrogenKind.THR
)
STANDARD_HYDROGEN_SEMANTICS["TYR"] = HydrogenSemantics(
    rotatable_kind=RotatableHydrogenKind.TYR
)
STANDARD_HYDROGEN_SEMANTICS["TRP"] = HydrogenSemantics(
    plan_with_backbone=STANDARD_HYDROGEN_PLANS["TRP"],
    plan_without_backbone=TRP_WITHOUT_BACKBONE_HYDROGEN_PLAN,
)


@lru_cache(maxsize=1)
def build_standard_component_library() -> ComponentLibrary:
    """Return the built-in library for the 20 standard residues."""

    templates: dict[str, ResidueTemplate] = {}
    for component_id, atom_names in STANDARD_COMPONENT_ATOMS.items():
        aliases = HISTIDINE_ALIASES if component_id == "HIS" else ()
        heavy_atom_semantics = STANDARD_HEAVY_ATOM_SEMANTICS.get(component_id)
        hydrogen_semantics = STANDARD_HYDROGEN_SEMANTICS.get(component_id)
        capabilities: set[ComponentCapability] = set()
        if heavy_atom_semantics is not None:
            capabilities.add(ComponentCapability.TEMPLATE_REPAIR)
        if hydrogen_semantics is not None:
            capabilities.add(ComponentCapability.HYDROGENATION)

        definition = ChemicalComponentDefinition(
            component_id=component_id,
            atom_names=atom_names,
            aliases=aliases,
            capabilities=frozenset(capabilities),
        )
        templates[component_id] = ResidueTemplate(
            definition=definition,
            forcefield_parameters={
                atom_name: ForceFieldAtomParams(
                    charge=params[0],
                    sigma_nm=params[1],
                    epsilon_kj_mol=params[2],
                )
                for atom_name, params in FORCEFIELD_PARAMETERS[component_id].items()
            },
            heavy_atom_semantics=heavy_atom_semantics,
            hydrogen_semantics=hydrogen_semantics,
        )

    return ComponentLibrary(templates=templates)
