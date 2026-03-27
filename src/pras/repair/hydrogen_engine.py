"""Hydrogen-placement engine over declarative residue semantics."""

from collections.abc import Mapping
from copy import deepcopy

from pras.chemistry.component import HydrogenSemantics, RotatableHydrogenKind
from pras.chemistry.hydrogen_plans import DISULFIDE_CYSTEINE_PLAN
from pras.repair.geometry import (
    RotatableHydrogenSearch,
    backbone_hydrogen,
    coordinate,
    cysteine_thiol,
    is_disulfide_bonded,
    n_terminal_hydrogens,
    optimize_rotatable_hydrogen,
    planar_pair,
    planar_single,
    serine_hydroxyl,
    tetrahedral_pair,
    tetrahedral_single,
    threonine_hydroxyl,
    torsion_angle,
    tyrosine_hydroxyl,
)
from pras.repair.payloads import (
    BackboneHydrogenPlacement,
    CoordinateBlock,
    HydrogenationContext,
    HydrogenPayloadResult,
    ResiduePayload,
)


def generate_hydrogen_payload(
    *,
    payload: ResiduePayload,
    context: HydrogenationContext,
    semantics: HydrogenSemantics,
) -> HydrogenPayloadResult:
    """Return the hydrogenated payload for one residue."""

    sidechain_atom_names: list[str]
    sidechain_coordinates: CoordinateBlock

    if semantics.rotatable_kind is RotatableHydrogenKind.CYS:
        sidechain_atom_names, sidechain_coordinates = cysteine_sidechain_hydrogens(
            payload=payload,
            context=context,
        )
    elif semantics.rotatable_kind is RotatableHydrogenKind.SER:
        sidechain_atom_names, sidechain_coordinates = serine_sidechain_hydrogens(
            payload=payload,
            context=context,
        )
    elif semantics.rotatable_kind is RotatableHydrogenKind.THR:
        sidechain_atom_names, sidechain_coordinates = threonine_sidechain_hydrogens(
            payload=payload,
            context=context,
        )
    elif semantics.rotatable_kind is RotatableHydrogenKind.TYR:
        sidechain_atom_names, sidechain_coordinates = tyrosine_sidechain_hydrogens(
            payload=payload,
            context=context,
        )
    else:
        sidechain_atom_names, sidechain_coordinates = standard_sidechain_hydrogens(
            payload=payload,
            semantics=semantics,
            include_backbone_hydrogen=context.include_backbone_hydrogen,
        )

    atom_names = deepcopy(payload.atom_names)
    atom_coordinates = deepcopy(payload.atom_coordinates)
    atom_names.extend(sidechain_atom_names)
    atom_coordinates.extend(sidechain_coordinates)

    if not context.include_backbone_hydrogen or context.next_payload is None:
        return HydrogenPayloadResult(
            atom_coordinates=atom_coordinates,
            atom_names=atom_names,
        )

    next_alpha_anchor, next_nitrogen_anchor = (
        context.next_payload.legacy_backbone_hydrogen_anchors()
    )
    propagated = backbone_hydrogen(
        next_alpha_anchor,
        next_nitrogen_anchor,
        payload.coordinate_map()["C"],
    )
    return HydrogenPayloadResult(
        atom_coordinates=atom_coordinates,
        atom_names=atom_names,
        backbone_hydrogen=BackboneHydrogenPlacement(
            coordinates=propagated,
            residue_index=context.residue_index,
        ),
    )


def standard_sidechain_hydrogens(
    *,
    payload: ResiduePayload,
    semantics: HydrogenSemantics,
    include_backbone_hydrogen: bool,
) -> tuple[list[str], CoordinateBlock]:
    """Return ordered sidechain hydrogens for a static residue plan."""

    plan = semantics.static_plan(
        include_backbone_hydrogen=include_backbone_hydrogen
    )
    if plan is None:
        raise ValueError("static hydrogen semantics require a plan")

    atom_coordinates = payload.coordinate_map()
    sidechain_atom_names: list[str] = []
    sidechain_coordinates: CoordinateBlock = []
    for output_names, method_name, arguments in plan:
        coordinates = evaluate_operation(
            method_name,
            arguments,
            atom_coordinates=atom_coordinates,
        )
        sidechain_atom_names.extend(output_names)
        sidechain_coordinates.extend(coordinates)

    return sidechain_atom_names, sidechain_coordinates


def cysteine_sidechain_hydrogens(
    *,
    payload: ResiduePayload,
    context: HydrogenationContext,
) -> tuple[list[str], CoordinateBlock]:
    """Return ordered sidechain hydrogens for a cysteine residue."""

    atom_coordinates = payload.coordinate_map()
    if is_disulfide_bonded(atom_coordinates["SG"], context.sg_coordinates):
        return evaluate_plan(DISULFIDE_CYSTEINE_PLAN, atom_coordinates)

    hydrogen = cysteine_thiol(
        atom_coordinates["SG"],
        atom_coordinates["CB"],
        atom_coordinates["CA"],
    )
    search = RotatableHydrogenSearch(
        outer_anchor=atom_coordinates["CA"],
        inner_anchor=atom_coordinates["CB"],
        donor=atom_coordinates["SG"],
        hydrogen=hydrogen,
        build_bond_length=1.34,
        reproject_bond_length=0.96,
        dihedral=torsion_angle(
            atom_coordinates["CA"],
            atom_coordinates["CB"],
            atom_coordinates["SG"],
            hydrogen,
        ),
        partial_charge=0.19,
        sigma=0.11,
        epsilon=0.07,
    )
    optimized = optimize_rotatable_hydrogen(
        context.residue_number,
        context.rotatable_hydrogen_environments,
        search,
    )
    return ["HA", "HB1", "HB2", "HG"], [
        tetrahedral_single(
            atom_coordinates["CB"],
            atom_coordinates["N"],
            atom_coordinates["CA"],
        ),
        *tetrahedral_pair(
            atom_coordinates["CA"],
            atom_coordinates["SG"],
            atom_coordinates["CB"],
        ),
        optimized,
    ]


def serine_sidechain_hydrogens(
    *,
    payload: ResiduePayload,
    context: HydrogenationContext,
) -> tuple[list[str], CoordinateBlock]:
    """Return ordered sidechain hydrogens for serine."""

    atom_coordinates = payload.coordinate_map()
    initial_hydrogen = serine_hydroxyl(
        atom_coordinates["OG"],
        atom_coordinates["CB"],
        atom_coordinates["CA"],
    )
    search = RotatableHydrogenSearch(
        outer_anchor=atom_coordinates["CA"],
        inner_anchor=atom_coordinates["CB"],
        donor=atom_coordinates["OG"],
        hydrogen=initial_hydrogen,
        build_bond_length=0.96,
        reproject_bond_length=0.96,
        dihedral=torsion_angle(
            atom_coordinates["CA"],
            atom_coordinates["CB"],
            atom_coordinates["OG"],
            initial_hydrogen,
        ),
        partial_charge=0.41,
        sigma=0.0,
        epsilon=0.0,
    )
    optimized = optimize_rotatable_hydrogen(
        context.residue_number,
        context.rotatable_hydrogen_environments,
        search,
    )
    hb1, hb2 = tetrahedral_pair(
        atom_coordinates["CA"],
        atom_coordinates["OG"],
        atom_coordinates["CB"],
    )
    return ["HA", "HB1", "HB2", "HG"], [
        tetrahedral_single(
            atom_coordinates["CB"],
            atom_coordinates["N"],
            atom_coordinates["CA"],
        ),
        hb1,
        hb2,
        optimized,
    ]


def threonine_sidechain_hydrogens(
    *,
    payload: ResiduePayload,
    context: HydrogenationContext,
) -> tuple[list[str], CoordinateBlock]:
    """Return ordered sidechain hydrogens for threonine."""

    atom_coordinates = payload.coordinate_map()
    initial_hydrogen = threonine_hydroxyl(
        atom_coordinates["OG1"],
        atom_coordinates["CB"],
        atom_coordinates["CG2"],
    )
    search = RotatableHydrogenSearch(
        outer_anchor=atom_coordinates["CA"],
        inner_anchor=atom_coordinates["CB"],
        donor=atom_coordinates["OG1"],
        hydrogen=initial_hydrogen,
        build_bond_length=0.96,
        reproject_bond_length=0.96,
        dihedral=torsion_angle(
            atom_coordinates["CA"],
            atom_coordinates["CB"],
            atom_coordinates["OG1"],
            initial_hydrogen,
        ),
        partial_charge=0.41,
        sigma=0.0,
        epsilon=0.0,
    )
    optimized = optimize_rotatable_hydrogen(
        context.residue_number,
        context.rotatable_hydrogen_environments,
        search,
    )
    return ["HG1", "HA", "HB", "1HG2", "2HG2", "3HG2"], [
        optimized,
        tetrahedral_single(
            atom_coordinates["CB"],
            atom_coordinates["N"],
            atom_coordinates["CA"],
        ),
        tetrahedral_single(
            atom_coordinates["CA"],
            atom_coordinates["OG1"],
            atom_coordinates["CB"],
        ),
        coordinate(
            atom_coordinates["OG1"],
            atom_coordinates["CB"],
            atom_coordinates["CG2"],
            1.09,
            60.5,
            109.4,
        ),
        coordinate(
            atom_coordinates["OG1"],
            atom_coordinates["CB"],
            atom_coordinates["CG2"],
            1.09,
            -179.5,
            109.5,
        ),
        coordinate(
            atom_coordinates["OG1"],
            atom_coordinates["CB"],
            atom_coordinates["CG2"],
            1.09,
            -59.5,
            109.5,
        ),
    ]


def tyrosine_sidechain_hydrogens(
    *,
    payload: ResiduePayload,
    context: HydrogenationContext,
) -> tuple[list[str], CoordinateBlock]:
    """Return ordered sidechain hydrogens for tyrosine."""

    atom_coordinates = payload.coordinate_map()
    initial_hydrogen = tyrosine_hydroxyl(
        atom_coordinates["OH"],
        atom_coordinates["CZ"],
        atom_coordinates["CE2"],
    )
    search = RotatableHydrogenSearch(
        outer_anchor=atom_coordinates["CE2"],
        inner_anchor=atom_coordinates["CZ"],
        donor=atom_coordinates["OH"],
        hydrogen=initial_hydrogen,
        build_bond_length=0.96,
        reproject_bond_length=0.96,
        dihedral=torsion_angle(
            atom_coordinates["CE2"],
            atom_coordinates["CZ"],
            atom_coordinates["OH"],
            initial_hydrogen,
        ),
        partial_charge=0.37,
        sigma=0.0,
        epsilon=0.0,
    )
    optimized = optimize_rotatable_hydrogen(
        context.residue_number,
        context.rotatable_hydrogen_environments,
        search,
    )
    hb1, hb2 = tetrahedral_pair(
        atom_coordinates["CA"],
        atom_coordinates["CG"],
        atom_coordinates["CB"],
    )
    return ["HA", "HB1", "HB2", "HD1", "HD2", "HE1", "HE2", "HH"], [
        tetrahedral_single(
            atom_coordinates["CB"],
            atom_coordinates["N"],
            atom_coordinates["CA"],
        ),
        hb1,
        hb2,
        planar_single(
            atom_coordinates["CG"],
            atom_coordinates["CD1"],
            atom_coordinates["CE1"],
            1.08,
        ),
        planar_single(
            atom_coordinates["CE2"],
            atom_coordinates["CD2"],
            atom_coordinates["CG"],
            1.08,
        ),
        planar_single(
            atom_coordinates["CZ"],
            atom_coordinates["CE1"],
            atom_coordinates["CD1"],
            1.08,
        ),
        planar_single(
            atom_coordinates["CZ"],
            atom_coordinates["CE2"],
            atom_coordinates["CD2"],
            1.08,
        ),
        optimized,
    ]


def histidine_delta_hydrogen(payload: ResiduePayload) -> list[float]:
    """Return the additional ND1 hydrogen used for protonated histidines."""

    atom_coordinates = payload.coordinate_map()
    return planar_single(
        atom_coordinates["CE1"],
        atom_coordinates["ND1"],
        atom_coordinates["CG"],
        1.01,
    )


def n_terminal_hydrogen_coordinates(
    payload: ResiduePayload,
    component_id: str,
) -> tuple[list[float], ...]:
    """Return the ordered N-terminal hydrogens for the first residue in a chain."""

    return n_terminal_hydrogens(component_id, payload.coordinate_map())


def evaluate_plan(
    plan: tuple[tuple[tuple[str, ...], str, tuple[str | float, ...]], ...],
    atom_coordinates: Mapping[str, list[float]],
) -> tuple[list[str], CoordinateBlock]:
    """Evaluate a declarative hydrogen plan against one residue."""

    atom_names: list[str] = []
    coordinates: CoordinateBlock = []
    for output_names, method_name, arguments in plan:
        coordinate_values = evaluate_operation(
            method_name,
            arguments,
            atom_coordinates=atom_coordinates,
        )
        atom_names.extend(output_names)
        coordinates.extend(coordinate_values)

    return atom_names, coordinates


def evaluate_operation(
    method_name: str,
    arguments: tuple[str | float, ...],
    *,
    atom_coordinates: Mapping[str, list[float]],
) -> CoordinateBlock:
    """Evaluate one geometry operation into one or more coordinates."""

    if method_name == "class2":
        bond_length = resolved_float(arguments[3]) if len(arguments) == 4 else 1.09
        first, second = tetrahedral_pair(
            resolved_coordinate(arguments[0], atom_coordinates),
            resolved_coordinate(arguments[1], atom_coordinates),
            resolved_coordinate(arguments[2], atom_coordinates),
            bond_length=bond_length,
        )
        return [first, second]

    if method_name == "class3":
        return [
            tetrahedral_single(
                resolved_coordinate(arguments[0], atom_coordinates),
                resolved_coordinate(arguments[1], atom_coordinates),
                resolved_coordinate(arguments[2], atom_coordinates),
            )
        ]

    if method_name == "class4":
        return [
            planar_pair(
                resolved_coordinate(arguments[0], atom_coordinates),
                resolved_coordinate(arguments[1], atom_coordinates),
                resolved_coordinate(arguments[2], atom_coordinates),
            )
        ]

    if method_name == "class5":
        return [
            planar_single(
                resolved_coordinate(arguments[0], atom_coordinates),
                resolved_coordinate(arguments[1], atom_coordinates),
                resolved_coordinate(arguments[2], atom_coordinates),
                resolved_float(arguments[3]),
            )
        ]

    if method_name == "calcCoordinate":
        return [
            coordinate(
                resolved_coordinate(arguments[0], atom_coordinates),
                resolved_coordinate(arguments[1], atom_coordinates),
                resolved_coordinate(arguments[2], atom_coordinates),
                resolved_float(arguments[3]),
                resolved_float(arguments[4]),
                resolved_float(arguments[5]),
            )
        ]

    raise ValueError(f"unsupported hydrogen geometry method {method_name!r}")


def resolved_coordinate(
    argument: str | float | int,
    atom_coordinates: Mapping[str, list[float]],
) -> list[float]:
    """Resolve a named atom argument to a coordinate."""

    if not isinstance(argument, str):
        raise TypeError("coordinate arguments must resolve to atom names")

    return atom_coordinates[argument]


def resolved_float(argument: str | float | int) -> float:
    """Resolve a numeric hydrogen-plan argument."""

    if isinstance(argument, str):
        raise TypeError("numeric hydrogen-plan arguments must be floats")

    return float(argument)
