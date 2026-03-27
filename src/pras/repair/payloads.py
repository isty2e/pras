"""Typed payloads and execution contexts for repair workflows."""

from dataclasses import dataclass

Coordinate = list[float]
CoordinateBlock = list[Coordinate]


@dataclass(frozen=True, slots=True)
class OrderedAtomPayload:
    """Ordered atom names with their coordinates."""

    atom_names: list[str]
    atom_coordinates: CoordinateBlock

    def coordinate_map(self) -> dict[str, Coordinate]:
        """Return a mutable coordinate mapping keyed by atom name."""

        return {
            atom_name: [float(value) for value in coordinates]
            for atom_name, coordinates in zip(
                self.atom_names,
                self.atom_coordinates,
                strict=True,
            )
        }


@dataclass(frozen=True, slots=True)
class ResiduePayload:
    """Execution payload for one residue during hydrogen placement."""

    residue_label: str
    atom_names: list[str]
    atom_coordinates: CoordinateBlock

    def coordinate_map(self) -> dict[str, Coordinate]:
        """Return a mutable coordinate mapping keyed by atom name."""

        return {
            atom_name: [float(value) for value in coordinates]
            for atom_name, coordinates in zip(
                self.atom_names,
                self.atom_coordinates,
                strict=True,
            )
        }

    def legacy_backbone_hydrogen_anchors(self) -> tuple[Coordinate, Coordinate]:
        """Return the next-residue anchors used by the legacy backbone-H rule.

        Upstream PRAS indexes the next residue by position instead of atom name when
        propagating the backbone amide hydrogen. We keep that quirk localized here so
        the execution layer can match legacy repair output without leaking positional
        semantics into the canonical residue model.
        """

        if len(self.atom_coordinates) < 2:
            raise ValueError(
                "backbone hydrogen propagation requires at least two atom coordinates"
            )

        return (
            [float(value) for value in self.atom_coordinates[1]],
            [float(value) for value in self.atom_coordinates[0]],
        )


@dataclass(frozen=True, slots=True)
class HeavyRepairContext:
    """Neighbor-dependent inputs required for heavy-atom repair."""

    next_residue_coordinates: CoordinateBlock
    psi_points: CoordinateBlock


@dataclass(frozen=True, slots=True)
class BackboneHydrogenPlacement:
    """Deferred backbone hydrogen propagated to the next residue."""

    coordinates: Coordinate
    residue_index: int


@dataclass(frozen=True, slots=True)
class HydrogenPayloadResult:
    """Hydrogenation result for one residue payload."""

    atom_coordinates: CoordinateBlock
    atom_names: list[str]
    backbone_hydrogen: BackboneHydrogenPlacement | None = None


@dataclass(frozen=True, slots=True)
class RotatableHydrogenEnvironment:
    """Packed heavy-atom interaction data for one residue-number environment."""

    residue_number: str
    atom_x: tuple[float, ...]
    atom_y: tuple[float, ...]
    atom_z: tuple[float, ...]
    charges: tuple[float, ...]
    sigmas_nm: tuple[float, ...]
    epsilons_kj_mol: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class HydrogenationContext:
    """Neighbor and chain context required for hydrogen placement."""

    residue_index: int
    residue_number: str
    rotatable_hydrogen_environments: tuple[RotatableHydrogenEnvironment, ...]
    sg_coordinates: CoordinateBlock
    next_payload: ResiduePayload | None
    include_backbone_hydrogen: bool
