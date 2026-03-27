"""Internal geometric primitives for protein repair workflows."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import acos, cos, degrees, pi, sin, sqrt

import numpy as np
from numpy.typing import NDArray

from pras.repair.payloads import RotatableHydrogenEnvironment

CoordinateLike = Sequence[float]
Vector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RotatableHydrogenSearch:
    """Inputs required to score a rotatable hydrogen over a 360-degree scan."""

    outer_anchor: CoordinateLike
    inner_anchor: CoordinateLike
    donor: CoordinateLike
    hydrogen: CoordinateLike
    build_bond_length: float
    reproject_bond_length: float
    dihedral: float
    partial_charge: float
    sigma: float
    epsilon: float


def coordinate(
    atom_a: CoordinateLike,
    atom_b: CoordinateLike,
    atom_c: CoordinateLike,
    bond_length: float,
    dihedral_angle: float,
    bond_angle: float,
) -> list[float]:
    """Build a fourth coordinate from three anchors and internal coordinates."""

    point_a = vector(atom_a)
    point_b = vector(atom_b)
    point_c = vector(atom_c)
    dihedral_radians = dihedral_angle * pi / 180.0

    axis_bc = point_c - point_b
    axis_ba = point_a - point_b
    projected = (
        axis_ba - (np.dot(axis_ba, axis_bc) / np.dot(axis_bc, axis_bc)) * axis_bc
    )
    perpendicular = np.cross(axis_bc, axis_ba)

    unit_projected = (projected / np.linalg.norm(projected)) * np.cos(dihedral_radians)
    unit_perpendicular = (perpendicular / np.linalg.norm(perpendicular)) * np.sin(
        dihedral_radians
    )
    temp_point = point_b + (unit_projected + unit_perpendicular)

    bond_cb = point_b - point_c
    bond_ct = temp_point - point_c
    angle_bct = acos(
        np.dot(bond_cb, bond_ct) / (np.linalg.norm(bond_cb) * np.linalg.norm(bond_ct))
    )
    rotate = bond_angle - degrees(angle_bct)

    normal = np.cross(bond_cb, bond_ct)
    unit_normal = normal / np.linalg.norm(normal)
    rotated = (
        point_c
        + bond_ct * np.cos(rotate * pi / 180.0)
        + np.cross(unit_normal, bond_ct) * np.sin(rotate * pi / 180.0)
        + unit_normal * np.dot(unit_normal, bond_ct) * (1 - np.cos(rotate * pi / 180.0))
    )

    scaled = (
        (rotated - point_c) * (bond_length / np.linalg.norm(rotated - point_c))
    ) + point_c
    return to_coordinate_list(scaled)


def heavy_coordinate(
    atom_a: CoordinateLike,
    atom_b: CoordinateLike,
    atom_c: CoordinateLike,
    bond_length: float,
    bond_angle: float,
    dihedral_angle: float,
) -> list[float]:
    """Build a heavy-atom coordinate using the heavy-repair parameter ordering."""

    return coordinate(atom_a, atom_b, atom_c, bond_length, dihedral_angle, bond_angle)


def torsion_angle(
    coord_1: CoordinateLike,
    coord_2: CoordinateLike,
    coord_3: CoordinateLike,
    coord_4: CoordinateLike,
) -> float:
    """Return the signed torsion angle defined by four points."""

    point_1 = vector(coord_1)
    point_2 = vector(coord_2)
    point_3 = vector(coord_3)
    point_4 = vector(coord_4)

    bond_12 = point_1 - point_2
    bond_32 = point_3 - point_2
    bond_43 = point_4 - point_3

    plane_13 = np.cross(bond_12, bond_32)
    plane_24 = np.cross(bond_43, bond_32)

    cosine = np.dot(plane_13, plane_24) / sqrt(
        np.dot(plane_13, plane_13) * np.dot(plane_24, plane_24)
    )
    clamped = min(1.0, max(-1.0, float(cosine)))
    angle = acos(clamped)

    if np.dot(plane_13, np.cross(plane_24, bond_32)) < 0:
        angle = -angle

    return degrees(angle)


def distance(coord_1: CoordinateLike, coord_2: CoordinateLike) -> float:
    """Return the Euclidean distance between two coordinates."""

    delta_x = float(coord_1[0]) - float(coord_2[0])
    delta_y = float(coord_1[1]) - float(coord_2[1])
    delta_z = float(coord_1[2]) - float(coord_2[2])
    return sqrt((delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z))


def tetrahedral_pair(
    atom_a: CoordinateLike,
    atom_b: CoordinateLike,
    center: CoordinateLike,
    bond_length: float = 1.09,
) -> tuple[list[float], list[float]]:
    """Return the two tetrahedral positions opposite two known vertices."""

    point_a = vector(atom_a)
    point_b = vector(atom_b)
    point_center = vector(center)

    midpoint = 0.5 * (point_a + point_b)
    reflected = midpoint + 2 * (point_center - midpoint)
    offset = np.cross(point_a - midpoint, point_center - midpoint) / np.linalg.norm(
        point_center - midpoint
    )

    candidate_1 = reflected - offset
    candidate_2 = reflected + offset
    return (
        scale_bond(point_center, candidate_1, bond_length),
        scale_bond(point_center, candidate_2, bond_length),
    )


def tetrahedral_single(
    atom_a: CoordinateLike,
    atom_b: CoordinateLike,
    center: CoordinateLike,
    bond_length: float = 1.09,
) -> list[float]:
    """Return the remaining tetrahedral position from three known vertices."""

    point_a = vector(atom_a)
    point_b = vector(atom_b)
    point_center = vector(center)

    midpoint = 0.5 * (point_a + point_b)
    reflected = midpoint + 2 * (point_center - midpoint)
    candidate = reflected + np.cross(
        point_a - midpoint, point_center - midpoint
    ) / np.linalg.norm(point_center - midpoint)
    return scale_bond(point_center, candidate, bond_length)


def planar_pair(
    atom_a: CoordinateLike,
    atom_b: CoordinateLike,
    center: CoordinateLike,
    bond_length: float = 1.01,
) -> list[float]:
    """Return one of the planar bisector hydrogens used in amides/guanidinium."""

    point_a = vector(atom_a)
    point_b = vector(atom_b)
    point_center = vector(center)
    candidate = ((point_a - point_b) + (point_center - point_b)) + point_center
    return scale_bond(point_center, candidate, bond_length)


def planar_single(
    atom_a: CoordinateLike,
    center: CoordinateLike,
    atom_b: CoordinateLike,
    bond_length: float,
) -> list[float]:
    """Return the planar hydrogen position used on aromatic or amide donors."""

    point_a = vector(atom_a)
    point_center = vector(center)
    point_b = vector(atom_b)
    candidate = ((point_center - point_a) + (point_center - point_b)) + point_center

    bond_b = point_b - point_center
    bond_candidate = candidate - point_center
    bond_a = point_a - point_center
    angle_b = acos(
        np.dot(bond_b, bond_candidate)
        / (np.linalg.norm(bond_b) * np.linalg.norm(bond_candidate))
    )
    average_angle = (
        degrees(
            acos(
                np.dot(bond_b, bond_candidate)
                / (np.linalg.norm(bond_b) * np.linalg.norm(bond_candidate))
            )
            + acos(
                np.dot(bond_a, bond_candidate)
                / (np.linalg.norm(bond_a) * np.linalg.norm(bond_candidate))
            )
        )
    ) / 2
    rotate = average_angle - degrees(angle_b)

    normal = np.cross(bond_b, bond_candidate)
    unit_normal = normal / np.linalg.norm(normal)
    rotated = (
        point_center
        + bond_candidate * np.cos(rotate * pi / 180.0)
        + np.cross(unit_normal, bond_candidate) * np.sin(rotate * pi / 180.0)
        + unit_normal
        * np.dot(unit_normal, bond_candidate)
        * (1 - np.cos(rotate * pi / 180.0))
    )
    return scale_bond(point_center, rotated, bond_length)


def backbone_hydrogen(
    next_alpha_carbon: CoordinateLike,
    next_nitrogen: CoordinateLike,
    carbonyl_carbon: CoordinateLike,
) -> list[float]:
    """Return the propagated backbone hydrogen placed on the next residue."""

    return planar_single(next_alpha_carbon, next_nitrogen, carbonyl_carbon, 1.01)


def hydroxyl_hydrogen(
    donor: CoordinateLike,
    atom_b: CoordinateLike,
    atom_c: CoordinateLike,
    *,
    rotation_degrees: float,
    bond_length: float,
) -> list[float]:
    """Return the pre-optimization coordinate for a rotatable hydroxyl/thiol H."""

    donor_vector = vector(donor)
    bond_vector = vector(atom_b) - donor_vector
    axis_vector = vector(atom_c) - vector(atom_b)
    rotated = rotate_about_axis(
        bond_vector,
        axis_vector,
        rotation_degrees * pi / 180.0,
    )
    candidate = rotated + donor_vector
    return scale_bond(donor_vector, candidate, bond_length)


def serine_hydroxyl(
    og: CoordinateLike,
    cb: CoordinateLike,
    ca: CoordinateLike,
) -> list[float]:
    """Return the initial SER hydroxyl hydrogen coordinate."""

    hb1, _ = tetrahedral_pair(ca, og, cb)
    return hydroxyl_hydrogen(og, cb, hb1, rotation_degrees=-240.2, bond_length=0.96)


def threonine_hydroxyl(
    og1: CoordinateLike,
    cb: CoordinateLike,
    cg2: CoordinateLike,
) -> list[float]:
    """Return the initial THR hydroxyl hydrogen coordinate."""

    return hydroxyl_hydrogen(og1, cb, cg2, rotation_degrees=-243.2, bond_length=0.96)


def cysteine_thiol(
    sg: CoordinateLike,
    cb: CoordinateLike,
    ca: CoordinateLike,
) -> list[float]:
    """Return the initial CYS thiol hydrogen coordinate."""

    return hydroxyl_hydrogen(sg, cb, ca, rotation_degrees=-243.2, bond_length=1.3)


def tyrosine_hydroxyl(
    oh: CoordinateLike,
    cz: CoordinateLike,
    ce2: CoordinateLike,
) -> list[float]:
    """Return the initial TYR hydroxyl hydrogen coordinate."""

    return hydroxyl_hydrogen(oh, cz, ce2, rotation_degrees=-220.2, bond_length=0.96)


def rotate_about_axis(vector_to_rotate: Vector, axis: Vector, theta: float) -> Vector:
    """Rotate a vector around an arbitrary axis by `theta` radians."""

    axis_length = np.linalg.norm(axis)
    x_norm = axis[0] / axis_length
    y_norm = axis[1] / axis_length
    z_norm = axis[2] / axis_length
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    one_minus_cos = 1.0 - cos_theta

    matrix = np.array(
        [
            [
                cos_theta + x_norm * x_norm * one_minus_cos,
                x_norm * y_norm * one_minus_cos - z_norm * sin_theta,
                x_norm * z_norm * one_minus_cos + y_norm * sin_theta,
            ],
            [
                x_norm * y_norm * one_minus_cos + z_norm * sin_theta,
                cos_theta + y_norm * y_norm * one_minus_cos,
                y_norm * z_norm * one_minus_cos - x_norm * sin_theta,
            ],
            [
                x_norm * z_norm * one_minus_cos - y_norm * sin_theta,
                y_norm * z_norm * one_minus_cos + x_norm * sin_theta,
                cos_theta + z_norm * z_norm * one_minus_cos,
            ],
        ],
        dtype=float,
    )
    return vector_to_rotate @ matrix


def optimize_rotatable_hydrogen(
    residue_number: str,
    environments: Sequence[RotatableHydrogenEnvironment],
    search: RotatableHydrogenSearch,
) -> list[float]:
    """Return the lowest-energy hydrogen coordinate from a six-step torsion scan."""

    candidate_hydrogens = rotated_hydrogen_candidates(search)
    total_energies: list[float] = []

    for environment in environments:
        if environment.residue_number != residue_number:
            continue

        for candidate in candidate_hydrogens:
            total_energies.append(
                hydrogen_potential_energy(candidate, environment, search)
            )

    if not total_energies:
        return to_coordinate_list(search.hydrogen)

    minimum_index = total_energies.index(min(total_energies))
    if minimum_index >= len(candidate_hydrogens):
        return to_coordinate_list(search.hydrogen)

    return candidate_hydrogens[minimum_index]


def rotated_hydrogen_candidates(search: RotatableHydrogenSearch) -> list[list[float]]:
    """Return the six candidate coordinates from the hydroxyl torsion scan."""

    candidates: list[list[float]] = []
    current_dihedral = search.dihedral
    for increment in range(0, 360, 60):
        # The legacy rotatable-H scan uses the potential-energy helper's
        # parameter ordering: dihedral first, scanned torsion second.
        candidate = coordinate(
            search.outer_anchor,
            search.inner_anchor,
            search.donor,
            search.build_bond_length,
            109.5,
            current_dihedral,
        )
        adjusted = recalculate_coordinate(
            search.inner_anchor,
            search.donor,
            candidate,
            search.reproject_bond_length,
        )
        candidates.append(adjusted)
        current_dihedral += increment

    return candidates


def hydrogen_potential_energy(
    hydrogen: CoordinateLike,
    environment: RotatableHydrogenEnvironment,
    search: RotatableHydrogenSearch,
) -> float:
    """Return the nonbonded energy between a candidate H and nearby heavy atoms."""

    total_energy = 0.0
    hydrogen_x = float(hydrogen[0])
    hydrogen_y = float(hydrogen[1])
    hydrogen_z = float(hydrogen[2])
    hydrogen_charge = search.partial_charge
    hydrogen_sigma = search.sigma
    hydrogen_epsilon = search.epsilon
    electrostatic_constant = 138.94

    for atom_x, atom_y, atom_z, atom_charge, atom_sigma, atom_epsilon in zip(
        environment.atom_x,
        environment.atom_y,
        environment.atom_z,
        environment.charges,
        environment.sigmas_nm,
        environment.epsilons_kj_mol,
        strict=True,
    ):
        delta_x = hydrogen_x - atom_x
        delta_y = hydrogen_y - atom_y
        delta_z = hydrogen_z - atom_z
        separation_sq_angstrom = (
            (delta_x * delta_x)
            + (delta_y * delta_y)
            + (delta_z * delta_z)
        )
        if separation_sq_angstrom > 6.25:
            continue

        separation = sqrt(separation_sq_angstrom) / 10.0
        coulomb = (
            electrostatic_constant * (hydrogen_charge * atom_charge)
        ) / separation
        mixed_epsilon = 4 * sqrt(hydrogen_epsilon * atom_epsilon)
        mixed_sigma = ((hydrogen_sigma + atom_sigma) * 0.5) / separation
        mixed_sigma_sq = mixed_sigma * mixed_sigma
        mixed_sigma_six = mixed_sigma_sq * mixed_sigma_sq * mixed_sigma_sq
        lennard_jones = (mixed_sigma_six * mixed_sigma_six) - mixed_sigma_six
        total_energy += coulomb + (mixed_epsilon * lennard_jones)

    return total_energy


def recalculate_coordinate(
    atom_b: CoordinateLike,
    atom_c: CoordinateLike,
    atom_d: CoordinateLike,
    bond_length: float,
) -> list[float]:
    """Re-normalize a rotated hydrogen coordinate to the expected bond length."""

    point_b = vector(atom_b)
    point_c = vector(atom_c)
    point_d = vector(atom_d)
    theta = 107.0

    bond_cb = point_b - point_c
    bond_dc = point_d - point_c
    angle_bcd = acos(
        np.dot(bond_cb, bond_dc) / (np.linalg.norm(bond_cb) * np.linalg.norm(bond_dc))
    )
    rotate = theta - degrees(angle_bcd)

    normal = np.cross(bond_cb, bond_dc)
    unit_normal = normal / np.linalg.norm(normal)
    rotated = (
        point_c
        + bond_dc * np.cos(rotate * pi / 180.0)
        + np.cross(unit_normal, bond_dc) * np.sin(rotate * pi / 180.0)
        + unit_normal * np.dot(unit_normal, bond_dc) * (1 - np.cos(rotate * pi / 180.0))
    )

    scaled = (
        (rotated - point_c) * (bond_length / np.linalg.norm(rotated - point_c))
    ) + point_c
    return to_coordinate_list(scaled)


def backbone_oxygen(
    nitrogen: CoordinateLike,
    alpha_carbon: CoordinateLike,
    carbonyl_carbon: CoordinateLike,
    clash_reference: CoordinateLike,
    psi_points: Sequence[CoordinateLike],
) -> list[float]:
    """Return the backbone oxygen coordinate chosen from the preferred dihedral set."""

    psi = torsion_angle(psi_points[0], psi_points[1], psi_points[2], psi_points[3])
    if psi >= 100:
        trial_dihedrals = (-30.0, 140.0, -140.0, 40.0)
    elif 0 < psi < 100:
        trial_dihedrals = (-140.0, 140.0, -30.0, 40.0)
    else:
        trial_dihedrals = (140.0, -140.0, -30.0, 40.0)

    candidate = heavy_coordinate(
        nitrogen,
        alpha_carbon,
        carbonyl_carbon,
        1.23,
        120.5,
        trial_dihedrals[0],
    )
    for dihedral in trial_dihedrals:
        candidate = heavy_coordinate(
            nitrogen,
            alpha_carbon,
            carbonyl_carbon,
            1.23,
            120.5,
            dihedral,
        )
        if distance(candidate, clash_reference) > 2.0:
            return candidate

    return candidate


def c_terminal_oxygen(
    nitrogen: CoordinateLike,
    alpha_carbon: CoordinateLike,
    carbonyl_carbon: CoordinateLike,
    backbone_oxygen_coordinate: CoordinateLike,
) -> list[float]:
    """Return the OXT coordinate from the current terminal peptide geometry."""

    dih = torsion_angle(
        nitrogen, alpha_carbon, carbonyl_carbon, backbone_oxygen_coordinate
    )
    return heavy_coordinate(
        nitrogen,
        alpha_carbon,
        carbonyl_carbon,
        1.25,
        122.5,
        180.0 + dih,
    )


def is_disulfide_bonded(
    sg_coordinate: CoordinateLike,
    all_sg_coordinates: Sequence[CoordinateLike],
) -> bool:
    """Return whether a cysteine sulfur is within disulfide-bond distance."""

    return any(
        0.0 < distance(sg_coordinate, candidate) <= 3.0
        for candidate in all_sg_coordinates
    )


def n_terminal_hydrogens(
    residue_name: str,
    atom_coordinates: Mapping[str, CoordinateLike],
) -> tuple[list[float], ...]:
    """Return the ordered N-terminal hydrogens for the first residue in a chain."""

    if residue_name == "PRO":
        return tetrahedral_pair(
            atom_coordinates["CA"],
            atom_coordinates["CD"],
            atom_coordinates["N"],
            bond_length=1.01,
        )

    reference = (
        atom_coordinates["C"] if residue_name == "GLY" else atom_coordinates["CB"]
    )
    return (
        coordinate(
            reference,
            atom_coordinates["CA"],
            atom_coordinates["N"],
            1.01,
            179.6,
            109.5,
        ),
        coordinate(
            reference,
            atom_coordinates["CA"],
            atom_coordinates["N"],
            1.01,
            -60.4,
            109.6,
        ),
        coordinate(
            reference,
            atom_coordinates["CA"],
            atom_coordinates["N"],
            1.01,
            60.2,
            109.6,
        ),
    )


def vector(coordinates: CoordinateLike) -> Vector:
    """Return a float vector from a coordinate-like sequence."""

    return np.asarray(coordinates, dtype=float)


def to_coordinate_list(coordinates: CoordinateLike) -> list[float]:
    """Return a plain three-float coordinate list."""

    return [float(value) for value in coordinates]


def scale_bond(origin: Vector, candidate: Vector, bond_length: float) -> list[float]:
    """Scale a candidate point to the desired bond length from the origin."""

    scaled = (
        (candidate - origin) * (bond_length / np.linalg.norm(candidate - origin))
    ) + origin
    return to_coordinate_list(scaled)
