"""gemmi-backed writers that project canonical structures to coordinate text."""

from pathlib import Path

from pras.io.normalization import (
    gemmi,
    infer_file_format,
    require_gemmi,
)
from pras.model import Atom, FileFormat, ProteinStructure, Residue


def write_structure(
    structure: ProteinStructure,
    output_path: Path,
    *,
    file_format: FileFormat | None = None,
) -> None:
    """Serialize a canonical structure to a coordinate file."""

    resolved_format = (
        infer_file_format(output_path) if file_format is None else file_format
    )

    output_path.write_text(
        write_structure_string(structure, resolved_format),
        encoding="utf-8",
    )


def write_structure_string(structure: ProteinStructure, file_format: FileFormat) -> str:
    """Serialize a canonical structure to a coordinate string."""

    raw_structure = build_gemmi_structure(structure)
    if file_format is FileFormat.PDB:
        return raw_structure.make_pdb_string()

    if file_format is FileFormat.MMCIF:
        return raw_structure.make_mmcif_document().as_string()

    raise ValueError(f"unsupported file format: {file_format}")


def build_gemmi_structure(structure: ProteinStructure):
    """Project the canonical structure model into a gemmi structure."""

    require_gemmi()
    assert gemmi is not None

    raw_structure = gemmi.Structure()
    raw_structure.name = structure.source_name or "pras"

    model = gemmi.Model(1)
    for chain_id, residues in residues_by_chain_id(structure):
        raw_chain = gemmi.Chain(chain_id)
        for residue in residues:
            raw_chain.add_residue(build_gemmi_residue(residue))

        model.add_chain(raw_chain)

    raw_structure.add_model(model)
    raw_structure.setup_entities()
    raw_structure.assign_label_seq_id()
    raw_structure.assign_subchains()
    raw_structure.assign_serial_numbers()
    return raw_structure


def build_gemmi_residue(residue: Residue):
    """Project a canonical residue into a gemmi residue."""

    assert gemmi is not None

    raw_residue = gemmi.Residue()
    raw_residue.name = residue.component_id
    raw_residue.seqid = gemmi.SeqId(
        residue.residue_id.seq_num,
        residue.residue_id.insertion_code or " ",
    )
    raw_residue.het_flag = "H" if residue.is_hetero else "A"
    raw_residue.entity_type = (
        gemmi.EntityType.NonPolymer if residue.is_hetero else gemmi.EntityType.Polymer
    )

    for atom in residue.atoms:
        raw_residue.add_atom(build_gemmi_atom(atom))

    return raw_residue


def build_gemmi_atom(atom: Atom):
    """Project a canonical atom into a gemmi atom."""

    assert gemmi is not None

    raw_atom = gemmi.Atom()
    raw_atom.name = atom.name
    raw_atom.altloc = "\x00" if atom.altloc is None else atom.altloc
    raw_atom.charge = 0 if atom.formal_charge is None else atom.formal_charge
    raw_atom.element = gemmi.Element(atom.element)
    raw_atom.pos = gemmi.Position(atom.position.x, atom.position.y, atom.position.z)
    raw_atom.occ = atom.occupancy
    raw_atom.b_iso = 0.0 if atom.b_factor is None else atom.b_factor
    return raw_atom


def residues_by_chain_id(
    structure: ProteinStructure,
) -> tuple[tuple[str, list[Residue]], ...]:
    """Return polymer and ligand residues grouped by chain in write order."""

    grouped_residues: dict[str, list[Residue]] = {
        chain.chain_id: list(chain.residues) for chain in structure.chains
    }
    polymer_chain_ids = tuple(chain.chain_id for chain in structure.chains)

    for ligand in structure.ligands:
        chain_id = ligand.residue_id.chain_id
        if chain_id not in grouped_residues:
            grouped_residues[chain_id] = []

        grouped_residues[chain_id].append(ligand)

    chain_order = polymer_chain_ids + tuple(
        chain_id
        for chain_id in grouped_residues
        if chain_id not in polymer_chain_ids
    )
    return tuple(
        (chain_id, grouped_residues[chain_id]) for chain_id in chain_order
    )
