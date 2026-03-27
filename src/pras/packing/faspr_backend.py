"""FASPR specialization of the generic side-chain packing backend seam."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from typing_extensions import Self

from pras.errors import PrasError, ResidueNotFoundError
from pras.io import read_structure_string, write_structure_string
from pras.model import (
    FileFormat,
    IssueSeverity,
    LigandPolicy,
    ProteinStructure,
    Residue,
    ValidationIssue,
    ValidationIssueKind,
)
from pras.packing.faspr_paths import (
    faspr_executable_path,
)
from pras.packing.plan import PackingAlphabet, PackingPlan
from pras.packing.types import (
    PackingCapabilities,
    PackingRequest,
    PackingResult,
)
from pras.process import ProcessOptions

BACKBONE_ATOM_NAMES: frozenset[str] = frozenset({"N", "CA", "C", "O"})
FASPR_CAPABILITIES = PackingCapabilities(
    supports_full_structure_packing=True,
    supports_local_packing=True,
    supports_partial_sequence=True,
    supports_refinement=False,
    supports_noncanonical_components=False,
    deterministic_given_same_inputs=True,
)
FASPR_ALPHABET = PackingAlphabet(
    {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
)


class PackingBackendError(PrasError):
    """Raised when a side-chain packing backend cannot satisfy one request."""


class PackingBackendExecutionError(PackingBackendError):
    """Raised when a side-chain packing subprocess fails."""


@dataclass(frozen=True, slots=True)
class FasprPackingBackend:
    """Subprocess-backed adapter for the packaged FASPR executable."""

    executable_path: Path | None = None

    def capabilities(self) -> PackingCapabilities:
        """Return the declared capability surface of the FASPR backend."""

        return FASPR_CAPABILITIES

    def pack(self, request: PackingRequest) -> PackingResult:
        """Execute one packing request through FASPR and return the packed model."""

        request.assert_supported_by(self.capabilities())
        plan = PackingPlan.from_request(request)
        execution_input = FasprExecutionInput.from_plan(plan)
        executable_path = resolve_faspr_executable_path(self.executable_path)
        validate_rotamer_library_near(executable_path)

        packed_structure = run_faspr(
            execution_input,
            executable_path=executable_path,
        )
        changed_residue_ids = plan.changed_residue_ids_after(packed_structure)
        issues = infer_packing_issues(plan, packed_structure)
        return PackingResult(
            packed_structure=packed_structure,
            changed_residue_ids=changed_residue_ids,
            issues=issues,
            backend_name="faspr",
            backend_version=None,
        )


@dataclass(frozen=True, slots=True)
class FasprExecutionInput:
    """Backend-specific execution input derived from a generic packing plan."""

    structure: ProteinStructure
    sequence_override: str | None = None

    @classmethod
    def from_plan(cls, plan: PackingPlan) -> Self:
        """Build one FASPR execution input from a generic packing plan."""

        prepared_structure = prepare_structure_for_faspr(plan)
        sequence_override = build_faspr_sequence_override(plan)
        return cls(
            structure=prepared_structure,
            sequence_override=sequence_override,
        )


def resolve_faspr_executable_path(executable_path: Path | None) -> Path:
    """Return a usable FASPR executable path."""

    resolved_path = (
        faspr_executable_path()
        if executable_path is None
        else executable_path
    )
    if not resolved_path.exists():
        raise FileNotFoundError(f"FASPR executable does not exist: {resolved_path}")

    if not resolved_path.is_file():
        raise PackingBackendError(
            f"FASPR executable path is not a file: {resolved_path}"
        )

    return resolved_path


def validate_rotamer_library_near(executable_path: Path) -> Path:
    """Return the expected rotamer-library path beside one FASPR executable."""

    sibling_library_path = executable_path.parent / "dun2010bbdep.bin"
    if not sibling_library_path.exists():
        raise PackingBackendError(
            "FASPR requires dun2010bbdep.bin to exist beside the executable"
        )

    return sibling_library_path


def prepare_structure_for_faspr(plan: PackingPlan) -> ProteinStructure:
    """Return one canonical structure compatible with FASPR expectations."""

    prepared_chains = []
    for chain in plan.structure.chains:
        prepared_residues = []
        for residue in chain.residues:
            validate_faspr_residue(residue)
            prepared_residues.append(strip_hydrogens_from_residue(residue))

        prepared_chains.append(
            type(chain)(
                chain_id=chain.chain_id,
                residues=tuple(prepared_residues),
            )
        )

    return ProteinStructure(
        chains=tuple(prepared_chains),
        ligands=plan.structure.ligands,
        source_format=plan.structure.source_format,
        source_name=plan.structure.source_name,
    )


def validate_faspr_residue(residue: Residue) -> None:
    """Raise when one residue cannot be represented in a FASPR request."""

    if residue.is_hetero:
        raise PackingBackendError(
            "FASPR does not support hetero residue "
            f"{residue.residue_id.display_token()}"
        )

    if not FASPR_ALPHABET.supports_component(residue.component_id):
        raise PackingBackendError(
            f"FASPR does not support component {residue.component_id}"
        )

    missing_backbone_atoms = tuple(
        atom_name
        for atom_name in BACKBONE_ATOM_NAMES
        if not residue.has_atom(atom_name)
    )
    if missing_backbone_atoms:
        raise PackingBackendError(
            f"FASPR requires complete backbone atoms for "
            f"{residue.residue_id.display_token()}: {', '.join(missing_backbone_atoms)}"
        )


def strip_hydrogens_from_residue(residue: Residue) -> Residue:
    """Return a residue with hydrogen atoms removed."""

    return residue.without_atoms(
        tuple(atom.name for atom in residue.atoms if atom.element == "H")
    )


def build_faspr_sequence_override(plan: PackingPlan) -> str | None:
    """Return one FASPR sequence override string when the request needs it."""

    if plan.spec.target_sequence is None and not plan.fixed_residue_ids():
        return None

    sequence_override = list(plan.effective_sequence_tokens(FASPR_ALPHABET))
    fixed_residue_ids = set(plan.fixed_residue_ids())
    for index, residue_id in enumerate(plan.polymer_residue_ids()):
        if residue_id in fixed_residue_ids:
            sequence_override[index] = sequence_override[index].lower()

    return "".join(sequence_override)


def run_faspr(
    execution_input: FasprExecutionInput,
    *,
    executable_path: Path,
) -> ProteinStructure:
    """Run FASPR on one prepared structure and return the packed result."""

    with TemporaryDirectory(prefix="pras-faspr-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path = temp_dir / "input.pdb"
        output_path = temp_dir / "output.pdb"
        input_path.write_text(
            write_structure_string(
                ProteinStructure(
                    chains=execution_input.structure.chains,
                    ligands=(),
                    source_format=FileFormat.PDB,
                    source_name=execution_input.structure.source_name,
                ),
                FileFormat.PDB,
            ),
            encoding="utf-8",
        )

        command = [str(executable_path), "-i", str(input_path), "-o", str(output_path)]
        if execution_input.sequence_override is not None:
            sequence_path = temp_dir / "sequence.txt"
            sequence_path.write_text(
                f"{execution_input.sequence_override}\n",
                encoding="utf-8",
            )
            command.extend(["-s", str(sequence_path)])

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=executable_path.parent,
        )
        if completed.returncode != 0:
            error_message = completed.stderr.strip() or completed.stdout.strip()
            raise PackingBackendExecutionError(
                "FASPR execution failed with "
                f"exit code {completed.returncode}: {error_message}"
            )

        if not output_path.exists():
            raise PackingBackendExecutionError(
                "FASPR completed without producing an output PDB"
            )

        packed_core = read_structure_string(
            output_path.read_text(encoding="utf-8"),
            FileFormat.PDB,
            options=ProcessOptions(ligand_policy=LigandPolicy.DROP),
            source_name=execution_input.structure.source_name,
        )

    return ProteinStructure(
        chains=packed_core.chains,
        ligands=execution_input.structure.ligands,
        source_format=execution_input.structure.source_format,
        source_name=execution_input.structure.source_name,
    )


def infer_packing_issues(
    plan: PackingPlan,
    packed: ProteinStructure,
) -> tuple[ValidationIssue, ...]:
    """Return structural warnings inferred from one packed structure."""

    issues: list[ValidationIssue] = []
    for residue in packed.iter_residues():
        try:
            plan.residue(residue.residue_id)
        except ResidueNotFoundError as error:
            raise PackingBackendExecutionError(
                "FASPR produced an unknown residue identifier"
            ) from error

        if residue.is_hetero:
            issues.append(
                ValidationIssue(
                    kind=ValidationIssueKind.UNSUPPORTED_COMPONENT,
                    severity=IssueSeverity.WARNING,
                    residue_id=residue.residue_id,
                    message=(
                        f"FASPR returned unexpected hetero residue "
                        f"{residue.residue_id.display_token()}"
                    ),
                )
            )

    return tuple(issues)
