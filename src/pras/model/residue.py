"""Residue-level domain entities for the redesigned PRAS package."""

from collections.abc import Collection, Iterable
from dataclasses import dataclass

from typing_extensions import Self

from pras.chemistry.component import ChemicalComponentDefinition
from pras.errors import AtomNotFoundError, ModelInvariantError
from pras.model.atom import Atom
from pras.model.enums import IssueSeverity, ValidationIssueKind
from pras.model.events import ValidationIssue
from pras.model.ids import ResidueId

BACKBONE_ATOM_NAMES: tuple[str, ...] = ("N", "CA", "C", "O", "OXT")


@dataclass(frozen=True, slots=True)
class Residue:
    """Canonical residue instance."""

    component_id: str
    residue_id: ResidueId
    atoms: tuple[Atom, ...]
    is_hetero: bool = False

    def __post_init__(self) -> None:
        component_id = self.component_id.strip().upper()
        atoms = tuple(self.atoms)
        atom_names = tuple(atom.name for atom in atoms)

        if not component_id:
            raise ValueError("component_id must not be blank")

        if len(atom_names) != len(set(atom_names)):
            raise ModelInvariantError(
                f"residue {self.residue_id.display_token()} contains duplicate "
                "atom names"
            )

        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "atoms", atoms)

    def atom_names(self) -> tuple[str, ...]:
        """Return atom names in residue order."""

        return tuple(atom.name for atom in self.atoms)

    def atom(self, atom_name: str) -> Atom:
        """Return a named atom or raise if it is missing."""

        normalized_name = atom_name.strip().upper()
        for atom in self.atoms:
            if atom.name == normalized_name:
                return atom

        raise AtomNotFoundError(
            f"{self.residue_id.display_token()} has no atom named {normalized_name}"
        )

    def has_atom(self, atom_name: str) -> bool:
        """Return whether a named atom is present."""

        normalized_name = atom_name.strip().upper()
        return normalized_name in self.atom_names()

    def backbone_atoms(self) -> tuple[Atom, ...]:
        """Return present backbone atoms in canonical order."""

        present_atoms = []
        for atom_name in BACKBONE_ATOM_NAMES:
            if self.has_atom(atom_name):
                present_atoms.append(self.atom(atom_name))

        return tuple(present_atoms)

    def missing_atoms(
        self, definition: ChemicalComponentDefinition
    ) -> tuple[str, ...]:
        """Return expected atoms that are missing from the residue."""

        present = set(self.atom_names())
        return tuple(
            atom_name
            for atom_name in definition.expected_atom_names()
            if atom_name not in present
        )

    def unexpected_atoms(
        self, definition: ChemicalComponentDefinition
    ) -> tuple[str, ...]:
        """Return residue atoms that are not part of the component definition."""

        expected = set(definition.expected_atom_names())
        return tuple(
            atom_name for atom_name in self.atom_names() if atom_name not in expected
        )

    def validate_against(
        self, definition: ChemicalComponentDefinition
    ) -> tuple[ValidationIssue, ...]:
        """Return structural issues relative to a component definition."""

        issues: list[ValidationIssue] = []
        missing = self.missing_atoms(definition)
        unexpected = self.unexpected_atoms(definition)

        if missing:
            issues.append(
                ValidationIssue(
                    kind=ValidationIssueKind.MISSING_EXPECTED_ATOMS,
                    severity=IssueSeverity.WARNING,
                    residue_id=self.residue_id,
                    message=(
                        f"{self.residue_id.display_token()} is missing atoms: "
                        f"{', '.join(missing)}"
                    ),
                )
            )

        if unexpected:
            issues.append(
                ValidationIssue(
                    kind=ValidationIssueKind.UNEXPECTED_ATOMS,
                    severity=IssueSeverity.WARNING,
                    residue_id=self.residue_id,
                    message=(
                        f"{self.residue_id.display_token()} contains unexpected atoms: "
                        f"{', '.join(unexpected)}"
                    ),
                )
            )

        backbone = {"N", "CA", "C"}
        if self.component_id == definition.component_id and not backbone.issubset(
            set(self.atom_names())
        ):
            issues.append(
                ValidationIssue(
                    kind=ValidationIssueKind.INVALID_BACKBONE,
                    severity=IssueSeverity.ERROR,
                    residue_id=self.residue_id,
                    message=(
                        f"{self.residue_id.display_token()} is missing required "
                        "backbone atoms"
                    ),
                )
            )

        return tuple(issues)

    def with_atom(self, atom: Atom) -> Self:
        """Return a copy with a single atom added or replaced by name."""

        updated_atoms = list(self.atoms)
        for index, current_atom in enumerate(updated_atoms):
            if current_atom.name == atom.name:
                updated_atoms[index] = atom
                return type(self)(
                    component_id=self.component_id,
                    residue_id=self.residue_id,
                    atoms=tuple(updated_atoms),
                    is_hetero=self.is_hetero,
                )

        updated_atoms.append(atom)
        return type(self)(
            component_id=self.component_id,
            residue_id=self.residue_id,
            atoms=tuple(updated_atoms),
            is_hetero=self.is_hetero,
        )

    def with_atoms(self, atoms: Iterable[Atom]) -> Self:
        """Return a copy with multiple atoms added or replaced by name."""

        residue = self
        for atom in atoms:
            residue = residue.with_atom(atom)

        return residue

    def without_atoms(self, atom_names: Collection[str]) -> Self:
        """Return a copy without a collection of named atoms."""

        names_to_remove = {atom_name.strip().upper() for atom_name in atom_names}
        kept_atoms = tuple(
            atom for atom in self.atoms if atom.name not in names_to_remove
        )
        return type(self)(
            component_id=self.component_id,
            residue_id=self.residue_id,
            atoms=kept_atoms,
            is_hetero=self.is_hetero,
        )
