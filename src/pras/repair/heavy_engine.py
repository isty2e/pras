"""Heavy-atom repair builders over canonical residue payloads."""

from collections.abc import Callable, Mapping, Sequence

from pras.chemistry.component import HeavyAtomBuilderKind, HeavyAtomSemantics
from pras.repair.geometry import (
    backbone_oxygen,
    c_terminal_oxygen,
    heavy_coordinate,
    torsion_angle,
)
from pras.repair.payloads import (
    HeavyRepairContext,
    OrderedAtomPayload,
)


def c_terminal_oxt(payload: OrderedAtomPayload) -> list[float]:
    """Return the terminal OXT coordinate from an ordered residue payload."""

    atom_map = payload.coordinate_map()
    return c_terminal_oxygen(
        atom_map["N"],
        atom_map["CA"],
        atom_map["C"],
        atom_map["O"],
    )


def repair_residue_payload(
    *,
    payload: OrderedAtomPayload,
    missing_atoms: list[str],
    context: HeavyRepairContext,
    semantics: HeavyAtomSemantics,
) -> OrderedAtomPayload:
    """Return repaired heavy-atom payload for one supported residue."""

    atom_map = payload.coordinate_map()
    repairer = HEAVY_REPAIRERS[semantics.builder_kind]
    repaired_map = repairer(atom_map, missing_atoms, context)
    return ordered_payload(semantics.atom_order, repaired_map)


def repair_ala(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair alanine backbone oxygen and beta carbon."""

    del missing_atoms
    return backbone_and_cb(atom_map, context, cb_dihedral=122.69)


def repair_gly(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair glycine backbone oxygen."""

    del missing_atoms
    atom_map["O"] = atom_map.get("O") or backbone_oxygen(
        atom_map["N"],
        atom_map["CA"],
        atom_map["C"],
        context.next_residue_coordinates[0],
        context.psi_points,
    )
    return atom_map


def repair_ser(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair serine beta carbon and hydroxyl oxygen."""

    del missing_atoms
    base = backbone_and_cb(atom_map, context, cb_dihedral=122.66)
    if "OG" not in base:
        base["OG"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.417,
            110.773,
            -63.3,
        )
    return base


def repair_asn(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair asparagine sidechain atoms."""

    base = backbone_and_cb(atom_map, context, cb_dihedral=123.23)
    if any(atom_name in {"CG", "OD1"} for atom_name in missing_atoms):
        base["CG"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.52,
            112.62,
            -65.5,
        )
        base["OD1"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.23,
            120.85,
            -58.3,
        )
        base["ND2"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.33,
            116.48,
            121.7,
        )
        return base

    od1_dihedral = torsion_angle(base["CA"], base["CB"], base["CG"], base["OD1"])
    base["ND2"] = heavy_coordinate(
        base["CA"],
        base["CB"],
        base["CG"],
        1.33,
        116.48,
        180.0 + od1_dihedral,
    )
    return base


def repair_asp(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair aspartate sidechain atoms."""

    base = backbone_and_cb(atom_map, context, cb_dihedral=122.82)
    if any(atom_name in {"CG", "OD1"} for atom_name in missing_atoms):
        base["CG"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.52,
            113.06,
            -66.4,
        )
        base["OD1"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.25,
            119.22,
            -46.7,
        )
        base["OD2"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.25,
            118.21,
            133.3,
        )
        return base

    od1_dihedral = torsion_angle(base["CA"], base["CB"], base["CG"], base["OD1"])
    base["OD2"] = heavy_coordinate(
        base["CA"],
        base["CB"],
        base["CG"],
        1.25,
        118.21,
        180.0 + od1_dihedral,
    )
    return base


def repair_cys(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair cysteine sidechain sulfur."""

    del missing_atoms
    base = backbone_and_cb(atom_map, context, cb_dihedral=122.50)
    if "SG" not in base:
        base["SG"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.81,
            113.82,
            -62.2,
        )
    return base


def repair_gln(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair glutamine sidechain atoms."""

    base = backbone_and_cb(atom_map, context, cb_dihedral=122.81)
    if any(atom_name in {"CG", "CD", "OE1"} for atom_name in missing_atoms):
        base["CG"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.52,
            113.75,
            -60.2,
        )
        base["CD"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.52,
            112.78,
            -69.6,
        )
        base["OE1"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD"],
            1.24,
            120.86,
            -50.5,
        )
        base["NE2"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD"],
            1.33,
            116.50,
            129.5,
        )
        return base

    oe1_dihedral = torsion_angle(base["CB"], base["CG"], base["CD"], base["OE1"])
    base["NE2"] = heavy_coordinate(
        base["CB"],
        base["CG"],
        base["CD"],
        1.33,
        116.50,
        180.0 + oe1_dihedral,
    )
    return base


def repair_glu(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair glutamate sidechain atoms."""

    base = backbone_and_cb(atom_map, context, cb_dihedral=122.87)
    if any(atom_name in {"CG", "CD", "OE1"} for atom_name in missing_atoms):
        base["CG"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.52,
            113.82,
            -63.8,
        )
        base["CD"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.52,
            119.02,
            -179.8,
        )
        base["OE1"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD"],
            1.25,
            119.02,
            -6.2,
        )
        base["OE2"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD"],
            1.25,
            118.08,
            173.8,
        )
        return base

    oe1_dihedral = torsion_angle(base["CB"], base["CG"], base["CD"], base["OE1"])
    base["OE2"] = heavy_coordinate(
        base["CB"],
        base["CG"],
        base["CD"],
        1.25,
        118.08,
        180.0 + oe1_dihedral,
    )
    return base


def repair_his(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair histidine imidazole atoms."""

    base = backbone_and_cb(atom_map, context, cb_dihedral=122.67)
    if any(atom_name in {"CG", "ND1"} for atom_name in missing_atoms):
        base["CG"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.49,
            113.74,
            -63.2,
        )
        base["ND1"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.38,
            122.85,
            -75.7,
        )
        base["CD2"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.35,
            130.61,
            104.3,
        )
        base["CE1"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["ND1"],
            1.32,
            108.5,
            180.0,
        )
        base["NE2"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD2"],
            1.35,
            108.5,
            180.0,
        )
        return base

    if "CD2" not in base:
        nd1_dihedral = torsion_angle(base["CA"], base["CB"], base["CG"], base["ND1"])
        base["CD2"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.35,
            130.61,
            180.0 + nd1_dihedral,
        )
    if "CE1" not in base:
        base["CE1"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["ND1"],
            1.32,
            108.5,
            180.0,
        )
    if "NE2" not in base:
        base["NE2"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD2"],
            1.35,
            108.5,
            180.0,
        )
    return base


def repair_lys(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair lysine sidechain atoms."""

    base = backbone_and_cb(atom_map, context, cb_dihedral=122.76)
    if any(atom_name in {"CG", "CD", "CE", "NZ"} for atom_name in missing_atoms):
        base["CG"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.52,
            113.83,
            -64.5,
        )
        base["CD"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.52,
            111.79,
            -178.1,
        )
        base["CE"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD"],
            1.46,
            111.68,
            -179.6,
        )
        base["NZ"] = heavy_coordinate(
            base["CG"],
            base["CD"],
            base["CE"],
            1.33,
            124.79,
            179.6,
        )
    return base


def repair_arg(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair arginine sidechain atoms."""

    base = backbone_and_cb(atom_map, context, cb_dihedral=122.76)
    if any(atom_name in {"CG", "CD", "NE", "CZ"} for atom_name in missing_atoms):
        base["CG"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.52,
            113.83,
            -65.2,
        )
        base["CD"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.52,
            111.79,
            -179.2,
        )
        base["NE"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD"],
            1.46,
            111.68,
            -179.3,
        )
        base["CZ"] = heavy_coordinate(
            base["CG"],
            base["CD"],
            base["NE"],
            1.33,
            124.79,
            -178.7,
        )
        base["NH1"] = heavy_coordinate(
            base["CD"],
            base["NE"],
            base["CZ"],
            1.33,
            120.64,
            0.0,
        )
        base["NH2"] = heavy_coordinate(
            base["CD"],
            base["NE"],
            base["CZ"],
            1.33,
            119.63,
            180.0,
        )
        return base

    if "NH1" not in base:
        base["NH1"] = heavy_coordinate(
            base["CD"],
            base["NE"],
            base["CZ"],
            1.33,
            120.64,
            0.0,
        )
    if "NH2" not in base:
        base["NH2"] = heavy_coordinate(
            base["CD"],
            base["NE"],
            base["CZ"],
            1.33,
            119.63,
            180.0,
        )
    return base


def repair_ile(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair isoleucine sidechain atoms."""

    base = backbone_and_cb(atom_map, context, cb_dihedral=123.23)
    if any(atom_name in {"CG1", "CD1"} for atom_name in missing_atoms):
        base["CG1"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.527,
            110.7,
            59.7,
        )
        base["CG2"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.527,
            110.4,
            -60.3,
        )
        base["CD1"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG1"],
            1.52,
            113.97,
            169.8,
        )
        return base

    cg1_dihedral = torsion_angle(base["N"], base["CA"], base["CB"], base["CG1"])
    base["CG2"] = heavy_coordinate(
        base["N"],
        base["CA"],
        base["CB"],
        1.527,
        110.4,
        cg1_dihedral - 120.0,
    )
    return base


def repair_leu(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair leucine sidechain atoms."""

    base = backbone_and_cb(atom_map, context, cb_dihedral=122.49)
    if any(atom_name in {"CG", "CD1"} for atom_name in missing_atoms):
        base["CG"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.53,
            116.10,
            -60.1,
        )
        base["CD1"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.524,
            112.50,
            174.9,
        )
        base["CD2"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.525,
            112.50,
            294.9,
        )
        return base

    cd1_dihedral = torsion_angle(base["CA"], base["CB"], base["CG"], base["CD1"])
    base["CD2"] = heavy_coordinate(
        base["CA"],
        base["CB"],
        base["CG"],
        1.525,
        112.50,
        cd1_dihedral + 120.0,
    )
    return base


def repair_met(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair methionine sidechain atoms."""

    base = backbone_and_cb(atom_map, context, cb_dihedral=122.67)
    if any(atom_name in {"CG", "SD", "CE"} for atom_name in missing_atoms):
        base["CG"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.52,
            113.68,
            -64.4,
        )
        base["SD"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.81,
            112.69,
            -179.6,
        )
        base["CE"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["SD"],
            1.79,
            100.61,
            70.1,
        )
    return base


def repair_phe(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair phenylalanine aromatic-ring atoms."""

    base = backbone_and_cb(atom_map, context, cb_dihedral=122.61)
    if any(atom_name in {"CG", "CD1"} for atom_name in missing_atoms):
        base["CG"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.50,
            113.85,
            -64.7,
        )
        base["CD1"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.39,
            120.0,
            93.3,
        )
        base["CD2"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.39,
            120.0,
            -86.7,
        )
        base["CE1"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD1"],
            1.39,
            120.0,
            180.0,
        )
        base["CE2"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD2"],
            1.39,
            120.0,
            180.0,
        )
        base["CZ"] = heavy_coordinate(
            base["CG"],
            base["CD1"],
            base["CE1"],
            1.39,
            120.0,
            0.0,
        )
        return base

    if "CD2" not in base:
        cd1_dihedral = torsion_angle(base["CA"], base["CB"], base["CG"], base["CD1"])
        base["CD2"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.39,
            120.0,
            cd1_dihedral - 180.0,
        )
    if "CE1" not in base:
        base["CE1"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD1"],
            1.39,
            120.0,
            180.0,
        )
    if "CE2" not in base:
        base["CE2"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD2"],
            1.39,
            120.0,
            180.0,
        )
    if "CZ" not in base:
        base["CZ"] = heavy_coordinate(
            base["CG"],
            base["CD1"],
            base["CE1"],
            1.39,
            120.0,
            0.0,
        )
    return base


def repair_pro(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair proline ring atoms."""

    base = backbone_and_cb(atom_map, context, cb_dihedral=115.30)
    if any(atom_name in {"CG", "CD"} for atom_name in missing_atoms):
        base["CG"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.49,
            104.21,
            29.6,
        )
        base["CD"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.50,
            105.03,
            -34.8,
        )
    return base


def repair_thr(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair threonine sidechain atoms."""

    base = backbone_and_cb(atom_map, context, cb_dihedral=123.10)
    if "OG1" not in base:
        base["OG1"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.43,
            109.18,
            60.0,
        )
        base["CG2"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.53,
            111.13,
            -60.0,
        )
        return base

    og1_dihedral = torsion_angle(base["N"], base["CA"], base["CB"], base["OG1"])
    base["CG2"] = heavy_coordinate(
        base["N"],
        base["CA"],
        base["CB"],
        1.53,
        111.13,
        og1_dihedral - 120.0,
    )
    return base


def repair_trp(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair tryptophan aromatic-ring atoms."""

    base = backbone_and_cb(atom_map, context, cb_dihedral=122.61)
    if any(atom_name in {"CG", "CD1"} for atom_name in missing_atoms):
        base["CG"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.50,
            114.10,
            -66.4,
        )
        base["CD1"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.37,
            127.07,
            96.3,
        )
        base["CD2"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.43,
            126.66,
            -83.7,
        )
        base["NE1"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD1"],
            1.38,
            108.5,
            180.0,
        )
        base["CE2"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD2"],
            1.40,
            108.5,
            180.0,
        )
        base["CE3"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD2"],
            1.40,
            133.83,
            0.0,
        )
        base["CZ2"] = heavy_coordinate(
            base["CG"],
            base["CD2"],
            base["CE2"],
            1.40,
            120.0,
            180.0,
        )
        base["CZ3"] = heavy_coordinate(
            base["CG"],
            base["CD2"],
            base["CE3"],
            1.40,
            120.0,
            180.0,
        )
        base["CH2"] = heavy_coordinate(
            base["CD2"],
            base["CE2"],
            base["CZ2"],
            1.40,
            120.0,
            0.0,
        )
        return base

    if "CD2" not in base:
        cd1_dihedral = torsion_angle(base["CA"], base["CB"], base["CG"], base["CD1"])
        base["CD2"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.43,
            126.66,
            cd1_dihedral - 180.0,
        )
    if "NE1" not in base:
        base["NE1"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD1"],
            1.38,
            108.5,
            180.0,
        )
    if "CE2" not in base:
        base["CE2"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD2"],
            1.40,
            108.5,
            180.0,
        )
    if "CE3" not in base:
        base["CE3"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD2"],
            1.40,
            133.83,
            0.0,
        )
    if "CZ2" not in base:
        base["CZ2"] = heavy_coordinate(
            base["CG"],
            base["CD2"],
            base["CE2"],
            1.40,
            120.0,
            180.0,
        )
    if "CZ3" not in base:
        base["CZ3"] = heavy_coordinate(
            base["CG"],
            base["CD2"],
            base["CE3"],
            1.40,
            120.0,
            180.0,
        )
    if "CH2" not in base:
        base["CH2"] = heavy_coordinate(
            base["CD2"],
            base["CE2"],
            base["CZ2"],
            1.40,
            120.0,
            0.0,
        )
    return base


def repair_tyr(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair tyrosine aromatic-ring atoms."""

    base = backbone_and_cb(atom_map, context, cb_dihedral=122.60)
    if any(atom_name in {"CG", "CD1"} for atom_name in missing_atoms):
        base["CG"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.51,
            113.8,
            -64.3,
        )
        base["CD1"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.39,
            120.98,
            93.1,
        )
        base["CD2"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.39,
            120.82,
            273.1,
        )
        base["CE1"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD1"],
            1.39,
            120.0,
            180.0,
        )
        base["CE2"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD2"],
            1.39,
            120.0,
            180.0,
        )
        base["CZ"] = heavy_coordinate(
            base["CG"],
            base["CD1"],
            base["CE1"],
            1.39,
            120.0,
            0.0,
        )
        base["OH"] = heavy_coordinate(
            base["CD1"],
            base["CE1"],
            base["CZ"],
            1.39,
            119.78,
            180.0,
        )
        return base

    if "CD2" not in base:
        cd1_dihedral = torsion_angle(base["CA"], base["CB"], base["CG"], base["CD1"])
        base["CD2"] = heavy_coordinate(
            base["CA"],
            base["CB"],
            base["CG"],
            1.39,
            120.82,
            cd1_dihedral + 180.0,
        )
    if "CE1" not in base:
        base["CE1"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD1"],
            1.39,
            120.0,
            180.0,
        )
    if "CE2" not in base:
        base["CE2"] = heavy_coordinate(
            base["CB"],
            base["CG"],
            base["CD2"],
            1.39,
            120.0,
            180.0,
        )
    if "CZ" not in base:
        base["CZ"] = heavy_coordinate(
            base["CG"],
            base["CD1"],
            base["CE1"],
            1.39,
            120.0,
            0.0,
        )
    if "OH" not in base:
        base["OH"] = heavy_coordinate(
            base["CD1"],
            base["CE1"],
            base["CZ"],
            1.39,
            119.78,
            180.0,
        )
    return base


def repair_val(
    atom_map: dict[str, list[float]],
    missing_atoms: Sequence[str],
    context: HeavyRepairContext,
) -> dict[str, list[float]]:
    """Repair valine sidechain atoms."""

    base = backbone_and_cb(atom_map, context, cb_dihedral=123.23)
    if "CG1" not in base:
        base["CG1"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.527,
            110.7,
            177.2,
        )
        base["CG2"] = heavy_coordinate(
            base["N"],
            base["CA"],
            base["CB"],
            1.527,
            110.4,
            297.2,
        )
        return base

    cg1_dihedral = torsion_angle(base["N"], base["CA"], base["CB"], base["CG1"])
    base["CG2"] = heavy_coordinate(
        base["N"],
        base["CA"],
        base["CB"],
        1.527,
        110.4,
        cg1_dihedral + 120.0,
    )
    return base


def backbone_and_cb(
    atom_map: dict[str, list[float]],
    context: HeavyRepairContext,
    *,
    cb_dihedral: float,
) -> dict[str, list[float]]:
    """Ensure backbone O and beta carbon are present for a residue."""

    atom_map["O"] = atom_map.get("O") or backbone_oxygen(
        atom_map["N"],
        atom_map["CA"],
        atom_map["C"],
        context.next_residue_coordinates[0],
        context.psi_points,
    )
    atom_map["CB"] = atom_map.get("CB") or heavy_coordinate(
        atom_map["N"],
        atom_map["C"],
        atom_map["CA"],
        1.52,
        109.5,
        cb_dihedral,
    )
    return atom_map


def ordered_payload(
    atom_order: Sequence[str],
    atom_map: Mapping[str, list[float]],
) -> OrderedAtomPayload:
    """Project a coordinate mapping back into deterministic payload order."""

    return OrderedAtomPayload(
        atom_names=list(atom_order),
        atom_coordinates=[atom_map[atom_name] for atom_name in atom_order],
    )


HeavyRepairer = Callable[
    [dict[str, list[float]], Sequence[str], HeavyRepairContext],
    dict[str, list[float]],
]

HEAVY_REPAIRERS: dict[HeavyAtomBuilderKind, HeavyRepairer] = {
    HeavyAtomBuilderKind.ALA: repair_ala,
    HeavyAtomBuilderKind.ARG: repair_arg,
    HeavyAtomBuilderKind.ASN: repair_asn,
    HeavyAtomBuilderKind.ASP: repair_asp,
    HeavyAtomBuilderKind.CYS: repair_cys,
    HeavyAtomBuilderKind.GLN: repair_gln,
    HeavyAtomBuilderKind.GLU: repair_glu,
    HeavyAtomBuilderKind.GLY: repair_gly,
    HeavyAtomBuilderKind.HIS: repair_his,
    HeavyAtomBuilderKind.ILE: repair_ile,
    HeavyAtomBuilderKind.LEU: repair_leu,
    HeavyAtomBuilderKind.LYS: repair_lys,
    HeavyAtomBuilderKind.MET: repair_met,
    HeavyAtomBuilderKind.PHE: repair_phe,
    HeavyAtomBuilderKind.PRO: repair_pro,
    HeavyAtomBuilderKind.SER: repair_ser,
    HeavyAtomBuilderKind.THR: repair_thr,
    HeavyAtomBuilderKind.TRP: repair_trp,
    HeavyAtomBuilderKind.TYR: repair_tyr,
    HeavyAtomBuilderKind.VAL: repair_val,
}
