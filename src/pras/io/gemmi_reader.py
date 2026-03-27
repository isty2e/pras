"""gemmi-backed readers that normalize directly into the canonical model."""

from pathlib import Path

from pras.io.normalization import (
    gemmi,
    infer_file_format,
    normalize_altloc,
    normalize_chain_id,
    normalize_formal_charge,
    normalize_insertion_code,
    require_gemmi,
    to_gemmi_coor_format,
)
from pras.model import (
    Atom,
    Chain,
    FileFormat,
    LigandPolicy,
    MutationPolicy,
    OccupancyPolicy,
    ProteinStructure,
    Residue,
    ResidueId,
    Vec3,
)
from pras.process import ProcessOptions


def read_structure(
    path: Path, options: ProcessOptions | None = None
) -> ProteinStructure:
    """Read a coordinate file into the canonical structure model."""

    require_gemmi()
    assert gemmi is not None

    file_format = infer_file_format(path)
    raw_structure = read_raw_structure(path, file_format)
    return normalize_structure(
        raw_structure,
        file_format=file_format,
        options=options,
        source_name=path.name,
    )


def read_structure_string(
    contents: str,
    file_format: FileFormat,
    *,
    options: ProcessOptions | None = None,
    source_name: str | None = None,
) -> ProteinStructure:
    """Read an in-memory coordinate payload into the canonical model."""

    require_gemmi()
    assert gemmi is not None

    raw_structure = read_raw_structure_string(contents, file_format)
    return normalize_structure(
        raw_structure,
        file_format=file_format,
        options=options,
        source_name=source_name,
    )


def normalize_structure(
    raw_structure,
    *,
    file_format: FileFormat,
    options: ProcessOptions | None = None,
    source_name: str | None = None,
) -> ProteinStructure:
    """Normalize a gemmi structure into the canonical domain model."""

    normalized_options = default_read_options() if options is None else options
    chains: list[tuple[str, list[Residue]]] = []
    ligands: list[Residue] = []

    if len(raw_structure) == 0:
        return ProteinStructure(
            chains=(),
            ligands=(),
            source_format=file_format,
            source_name=source_name,
        )

    model = raw_structure[0]
    for raw_chain in model:
        chain_id = normalize_chain_id(raw_chain.name)
        if not normalized_options.selects_chain(chain_id):
            continue

        polymer_residues = normalize_polymer_residues(
            raw_chain, chain_id, normalized_options
        )
        chain_ligands = normalize_ligands(raw_chain, chain_id, normalized_options)
        if polymer_residues:
            chains.append((chain_id, polymer_residues))
        ligands.extend(chain_ligands)

    return ProteinStructure(
        chains=tuple(
            build_chain(chain_id, residues) for chain_id, residues in chains
        ),
        ligands=tuple(ligands),
        source_format=file_format,
        source_name=source_name,
    )


def default_read_options() -> ProcessOptions:
    """Return default ingress options that preserve source information."""

    return ProcessOptions(ligand_policy=LigandPolicy.KEEP)


def read_raw_structure(path: Path, file_format: FileFormat):
    """Read one coordinate file with a format-specific gemmi ingress path."""

    assert gemmi is not None

    if file_format is FileFormat.PDB:
        return gemmi.read_pdb(str(path))

    return gemmi.read_structure(
        str(path),
        format=to_gemmi_coor_format(file_format),
    )


def read_raw_structure_string(contents: str, file_format: FileFormat):
    """Read one coordinate payload with a format-specific gemmi ingress path."""

    assert gemmi is not None

    if file_format is FileFormat.PDB:
        return gemmi.read_pdb_string(contents)

    return gemmi.read_structure_string(
        contents,
        True,
        to_gemmi_coor_format(file_format),
    )


def normalize_polymer_residues(
    raw_chain, chain_id: str, options: ProcessOptions
) -> list[Residue]:
    """Normalize polymer residues in a chain, resolving duplicate residue ids."""

    grouped_residues: dict[ResidueId, list[Residue]] = {}
    residue_order: list[ResidueId] = []

    for raw_residue in raw_chain:
        if is_water_residue(raw_residue) or is_ligand_residue(raw_residue):
            continue

        residue = normalize_residue(raw_residue, chain_id, options.occupancy_policy)
        residue_id = residue.residue_id
        if residue_id not in grouped_residues:
            grouped_residues[residue_id] = []
            residue_order.append(residue_id)

        grouped_residues[residue_id].append(residue)

    return [
        select_residue_variant(grouped_residues[residue_id], options.mutation_policy)
        for residue_id in residue_order
    ]


def normalize_ligands(
    raw_chain, chain_id: str, options: ProcessOptions
) -> list[Residue]:
    """Normalize ligand residues in a chain according to the ligand policy."""

    if options.ligand_policy is LigandPolicy.DROP:
        return []

    ligands: list[Residue] = []
    for raw_residue in raw_chain:
        if not is_ligand_residue(raw_residue):
            continue

        ligands.append(
            normalize_residue(raw_residue, chain_id, options.occupancy_policy)
        )

    return ligands


def normalize_residue(
    raw_residue, chain_id: str, occupancy_policy: OccupancyPolicy
) -> Residue:
    """Normalize one gemmi residue into the canonical residue entity."""

    atoms = select_atom_variants(raw_residue, occupancy_policy)
    return Residue(
        component_id=raw_residue.name,
        residue_id=ResidueId(
            chain_id=chain_id,
            seq_num=int(raw_residue.seqid.num),
            insertion_code=normalize_insertion_code(raw_residue.seqid.icode),
        ),
        atoms=tuple(atoms),
        is_hetero=is_ligand_residue(raw_residue),
    )


def select_atom_variants(raw_residue, occupancy_policy: OccupancyPolicy) -> list[Atom]:
    """Resolve duplicate atom sites by atom name using the occupancy policy."""

    selected_atoms = {}
    atom_order: list[str] = []

    for raw_atom in raw_residue:
        atom_name = raw_atom.name.strip().upper()
        if atom_name not in selected_atoms:
            selected_atoms[atom_name] = raw_atom
            atom_order.append(atom_name)
            continue

        current_atom = selected_atoms[atom_name]
        if should_replace_atom(
            float(current_atom.occ), float(raw_atom.occ), occupancy_policy
        ):
            selected_atoms[atom_name] = raw_atom

    return [atom_from_raw_site(selected_atoms[atom_name]) for atom_name in atom_order]


def should_replace_atom(
    current_occupancy: float,
    candidate_occupancy: float,
    occupancy_policy: OccupancyPolicy,
) -> bool:
    """Decide whether a candidate atom site should replace the current choice."""

    if occupancy_policy is OccupancyPolicy.LOWEST:
        return candidate_occupancy < current_occupancy

    return candidate_occupancy > current_occupancy


def atom_from_raw_site(raw_atom) -> Atom:
    """Project one selected gemmi atom site into the canonical atom model."""

    return Atom(
        name=raw_atom.name,
        element=raw_atom.element.name,
        position=Vec3(
            x=float(raw_atom.pos.x),
            y=float(raw_atom.pos.y),
            z=float(raw_atom.pos.z),
        ),
        occupancy=float(raw_atom.occ),
        b_factor=float(raw_atom.b_iso),
        formal_charge=normalize_formal_charge(int(raw_atom.charge)),
        altloc=normalize_altloc(raw_atom.altloc),
    )


def select_residue_variant(
    residues: list[Residue], mutation_policy: MutationPolicy
) -> Residue:
    """Resolve duplicate residue ids produced by microheterogeneity."""

    if len(residues) == 1:
        return residues[0]

    best_residue = residues[0]
    for residue in residues[1:]:
        if should_replace_residue(best_residue, residue, mutation_policy):
            best_residue = residue

    return best_residue


def should_replace_residue(
    current_residue: Residue,
    candidate_residue: Residue,
    mutation_policy: MutationPolicy,
) -> bool:
    """Decide whether a residue variant should replace the current choice."""

    current_score = residue_occupancy_score(current_residue)
    candidate_score = residue_occupancy_score(candidate_residue)
    if mutation_policy is MutationPolicy.LOWEST_OCCUPANCY:
        return candidate_score < current_score

    return candidate_score > current_score


def residue_occupancy_score(residue: Residue) -> float:
    """Aggregate atom occupancies for residue-variant comparison."""

    return sum(atom.occupancy for atom in residue.atoms)


def is_ligand_residue(raw_residue) -> bool:
    """Return whether a gemmi residue should be classified as a non-water ligand."""

    require_gemmi()
    assert gemmi is not None

    return bool(
        not is_water_residue(raw_residue)
        and (
            raw_residue.het_flag == "H"
            or raw_residue.entity_type is gemmi.EntityType.NonPolymer
        )
    )


def is_water_residue(raw_residue) -> bool:
    """Return whether a gemmi residue represents water."""

    require_gemmi()
    assert gemmi is not None

    return bool(
        raw_residue.is_water()
        or raw_residue.entity_type is gemmi.EntityType.Water
    )


def build_chain(chain_id: str, residues: list[Residue]) -> Chain:
    """Build a canonical chain from normalized residues."""

    return Chain(chain_id=chain_id, residues=tuple(residues))
