"""Chemical component definitions for the redesigned PRAS package."""

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


class ComponentCapability(str, Enum):
    """Capabilities advertised by a chemical component definition."""

    TEMPLATE_REPAIR = "template_repair"
    HYDROGENATION = "hydrogenation"
    RDKIT_REFINEMENT = "rdkit_refinement"


class HeavyAtomBuilderKind(str, Enum):
    """Heavy-atom builder variants supported by the standard template set."""

    ALA = "ALA"
    ARG = "ARG"
    ASN = "ASN"
    ASP = "ASP"
    CYS = "CYS"
    GLN = "GLN"
    GLU = "GLU"
    GLY = "GLY"
    HIS = "HIS"
    ILE = "ILE"
    LEU = "LEU"
    LYS = "LYS"
    MET = "MET"
    PHE = "PHE"
    PRO = "PRO"
    SER = "SER"
    THR = "THR"
    TRP = "TRP"
    TYR = "TYR"
    VAL = "VAL"


class RotatableHydrogenKind(str, Enum):
    """Rotatable sidechain hydrogen families."""

    CYS = "CYS"
    SER = "SER"
    THR = "THR"
    TYR = "TYR"


HydrogenPlanArgument = str | float | int
HydrogenOperation = tuple[tuple[str, ...], str, tuple[HydrogenPlanArgument, ...]]


@dataclass(frozen=True, slots=True)
class BondDefinition:
    """Bond relationship between two atoms in a component definition."""

    atom_name_1: str
    atom_name_2: str
    order: int = 1
    aromatic: bool = False

    def __post_init__(self) -> None:
        atom_name_1 = self.atom_name_1.strip().upper()
        atom_name_2 = self.atom_name_2.strip().upper()
        if not atom_name_1 or not atom_name_2:
            raise ValueError("bond atom names must not be blank")

        if self.order <= 0:
            raise ValueError("bond order must be positive")

        object.__setattr__(self, "atom_name_1", atom_name_1)
        object.__setattr__(self, "atom_name_2", atom_name_2)


@dataclass(frozen=True, slots=True)
class ForceFieldAtomParams:
    """Per-atom force-field parameters used by residue templates."""

    charge: float
    sigma_nm: float
    epsilon_kj_mol: float


@dataclass(frozen=True, slots=True)
class HeavyAtomSemantics:
    """Heavy-atom repair semantics attached to a residue template."""

    builder_kind: HeavyAtomBuilderKind
    atom_order: tuple[str, ...]

    def __post_init__(self) -> None:
        atom_order = tuple(atom_name.strip().upper() for atom_name in self.atom_order)
        if not atom_order:
            raise ValueError("heavy-atom semantics require at least one atom name")

        if any(not atom_name for atom_name in atom_order):
            raise ValueError("heavy-atom semantics atom names must not be blank")

        if len(atom_order) != len(set(atom_order)):
            raise ValueError("heavy-atom semantics atom names must be unique")

        object.__setattr__(self, "atom_order", atom_order)


@dataclass(frozen=True, slots=True)
class HydrogenSemantics:
    """Hydrogen-placement semantics attached to a residue template."""

    plan_with_backbone: tuple[HydrogenOperation, ...] | None = None
    plan_without_backbone: tuple[HydrogenOperation, ...] | None = None
    rotatable_kind: RotatableHydrogenKind | None = None

    def __post_init__(self) -> None:
        uses_static_plan = (
            self.plan_with_backbone is not None
            or self.plan_without_backbone is not None
        )
        uses_rotatable_plan = self.rotatable_kind is not None

        if uses_static_plan and uses_rotatable_plan:
            raise ValueError(
                "hydrogen semantics must use either static plans or a rotatable kind"
            )

        if not uses_static_plan and not uses_rotatable_plan:
            raise ValueError(
                "hydrogen semantics require a static plan or a rotatable kind"
            )

        if self.plan_without_backbone is not None and self.plan_with_backbone is None:
            raise ValueError(
                "hydrogen semantics with_backbone plan is required when "
                "plan_without_backbone is provided"
            )

    def static_plan(
        self, *, include_backbone_hydrogen: bool
    ) -> tuple[HydrogenOperation, ...] | None:
        """Return the active static plan for a residue context."""

        if not include_backbone_hydrogen and self.plan_without_backbone is not None:
            return self.plan_without_backbone

        return self.plan_with_backbone


@dataclass(frozen=True, slots=True)
class ChemicalComponentDefinition:
    """Chemical definition for a residue or ligand-like component."""

    component_id: str
    atom_names: tuple[str, ...]
    bonds: tuple[BondDefinition, ...] = ()
    formal_charges: Mapping[str, int] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    capabilities: frozenset[ComponentCapability] = frozenset()

    def __post_init__(self) -> None:
        component_id = self.component_id.strip().upper()
        atom_names = tuple(atom_name.strip().upper() for atom_name in self.atom_names)
        aliases = tuple(alias.strip().upper() for alias in self.aliases)

        if not component_id:
            raise ValueError("component_id must not be blank")

        if not atom_names:
            raise ValueError("component definitions must contain at least one atom")

        if len(atom_names) != len(set(atom_names)):
            raise ValueError("component atom names must be unique")

        charges = {
            atom_name.strip().upper(): int(charge)
            for atom_name, charge in self.formal_charges.items()
        }

        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "atom_names", atom_names)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "formal_charges", MappingProxyType(charges))

    def expected_atom_names(self) -> tuple[str, ...]:
        """Return the canonical atom order for the component."""

        return self.atom_names

    def has_atom(self, atom_name: str) -> bool:
        """Return whether the definition contains a named atom."""

        return atom_name.strip().upper() in self.atom_names

    def supports(self, capability: ComponentCapability) -> bool:
        """Return whether a capability is advertised by this component."""

        return capability in self.capabilities

    def all_component_ids(self) -> tuple[str, ...]:
        """Return the canonical identifier plus all aliases."""

        return (self.component_id, *self.aliases)


@dataclass(frozen=True, slots=True)
class ResidueTemplate:
    """Residue template combining a chemical definition with force-field data."""

    definition: ChemicalComponentDefinition
    forcefield_parameters: Mapping[str, ForceFieldAtomParams] = field(
        default_factory=dict
    )
    preferred_atom_order: tuple[str, ...] = ()
    heavy_atom_semantics: HeavyAtomSemantics | None = None
    hydrogen_semantics: HydrogenSemantics | None = None

    def __post_init__(self) -> None:
        normalized_params = {
            atom_name.strip().upper(): params
            for atom_name, params in self.forcefield_parameters.items()
        }
        preferred_atom_order = tuple(
            atom_name.strip().upper() for atom_name in self.preferred_atom_order
        )

        object.__setattr__(
            self, "forcefield_parameters", MappingProxyType(normalized_params)
        )
        object.__setattr__(self, "preferred_atom_order", preferred_atom_order)

    def ordered_atom_names(self) -> tuple[str, ...]:
        """Return the preferred atom order for serialization or comparison."""

        return self.preferred_atom_order or self.definition.expected_atom_names()

    @property
    def component_id(self) -> str:
        """Return the canonical component identifier for the template."""

        return self.definition.component_id

    @property
    def aliases(self) -> tuple[str, ...]:
        """Return all aliases accepted for this template."""

        return self.definition.aliases

    def expected_atom_names(self) -> tuple[str, ...]:
        """Return the expected canonical atom order for the component."""

        return self.definition.expected_atom_names()

    def missing_atom_names(
        self,
        present_atom_names: Collection[str],
        *,
        exclude_atom_names: Collection[str] = (),
    ) -> tuple[str, ...]:
        """Return missing template atoms for a given present-atom set."""

        present = {atom_name.strip().upper() for atom_name in present_atom_names}
        excluded = {atom_name.strip().upper() for atom_name in exclude_atom_names}
        return tuple(
            atom_name
            for atom_name in self.expected_atom_names()
            if atom_name not in present and atom_name not in excluded
        )

    def supports(self, capability: ComponentCapability) -> bool:
        """Return whether a capability is advertised by this template."""

        return self.definition.supports(capability)

    def can_repair_heavy_atoms(self) -> bool:
        """Return whether heavy-atom repair semantics are available."""

        return self.heavy_atom_semantics is not None

    def can_add_hydrogens(self) -> bool:
        """Return whether hydrogen-placement semantics are available."""

        return self.hydrogen_semantics is not None

    def has_forcefield_params(self, atom_name: str) -> bool:
        """Return whether force-field parameters exist for a named atom."""

        return atom_name.strip().upper() in self.forcefield_parameters
