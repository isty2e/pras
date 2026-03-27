"""Atom-level domain entities for the redesigned PRAS package."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self


@dataclass(frozen=True, slots=True)
class Vec3:
    """Immutable 3D coordinate value object."""

    x: float
    y: float
    z: float

    @classmethod
    def from_iterable(cls, coordinates: Iterable[float]) -> Self:
        """Construct a coordinate from a length-3 iterable."""

        values = tuple(float(value) for value in coordinates)
        if len(values) != 3:
            raise ValueError("Vec3 requires exactly three coordinates")

        return cls(x=values[0], y=values[1], z=values[2])

    def to_array(self) -> NDArray[np.float64]:
        """Convert the coordinate to a NumPy vector."""

        return np.asarray((self.x, self.y, self.z), dtype=np.float64)

    def distance_to(self, other: Self) -> float:
        """Return the Euclidean distance to another coordinate."""

        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return sqrt(dx * dx + dy * dy + dz * dz)

    def with_offset(self, dx: float, dy: float, dz: float) -> Self:
        """Return a translated coordinate."""

        return type(self)(x=self.x + dx, y=self.y + dy, z=self.z + dz)

    def __iter__(self) -> Iterator[float]:
        """Yield coordinates in Cartesian order."""

        yield self.x
        yield self.y
        yield self.z


@dataclass(frozen=True, slots=True)
class Atom:
    """Canonical atom instance."""

    name: str
    element: str
    position: Vec3
    occupancy: float = 1.0
    b_factor: float | None = None
    formal_charge: int | None = None
    altloc: str | None = None

    def __post_init__(self) -> None:
        name = self.name.strip().upper()
        element = self.element.strip().upper()
        altloc = self.altloc
        if altloc is not None:
            altloc = altloc.strip() or None

        if not name:
            raise ValueError("atom name must not be blank")

        if not element:
            raise ValueError("atom element must not be blank")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "element", element)
        object.__setattr__(self, "altloc", altloc)

    def is_named(self, atom_name: str) -> bool:
        """Return whether this atom matches the requested name."""

        return self.name == atom_name.strip().upper()

    def distance_to(self, other: Self) -> float:
        """Return the distance to another atom."""

        return self.position.distance_to(other.position)

    def with_position(self, position: Vec3) -> Self:
        """Return a copy with updated coordinates."""

        return type(self)(
            name=self.name,
            element=self.element,
            position=position,
            occupancy=self.occupancy,
            b_factor=self.b_factor,
            formal_charge=self.formal_charge,
            altloc=self.altloc,
        )
