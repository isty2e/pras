"""Permanent representative regression fixtures for repair and workflow tests."""

from dataclasses import dataclass
from pathlib import Path

from tests.support.structure_summary import ChainSummary, StructureSummary


@dataclass(frozen=True, slots=True)
class RepresentativeCase:
    """Representative input fixture and expected canonical summary."""

    input_path: Path
    summary: StructureSummary
    protonate_histidines: bool = False


REPRESENTATIVE_CASES: dict[str, RepresentativeCase] = {
    "1aho-heavy-default": RepresentativeCase(
        input_path=Path("tests/fixtures/pdb/1aho.pdb"),
        summary=StructureSummary(
            chain_ids=("A",),
            chain_summaries=(ChainSummary("A", 64, 505),),
            residue_count=64,
            ligand_count=0,
            atom_count=505,
            hydrogen_atom_count=0,
            component_counts=tuple(
                sorted(
                    {
                        "ALA": 3,
                        "ARG": 3,
                        "ASN": 4,
                        "ASP": 4,
                        "CYS": 8,
                        "GLN": 1,
                        "GLU": 3,
                        "GLY": 7,
                        "HIS": 2,
                        "ILE": 1,
                        "LEU": 2,
                        "LYS": 5,
                        "PHE": 1,
                        "PRO": 3,
                        "SER": 2,
                        "THR": 3,
                        "TRP": 1,
                        "TYR": 7,
                        "VAL": 4,
                    }.items()
                )
            ),
            first_residue="ATOM:VAL:A:1",
            last_residue="ATOM:HIS:A:64",
            semantic_digest="dd27115aeb547300f639623930d1da438d64f386d6bc1153d986edad3ae0c378",
        ),
    ),
    "1aho-hydrogen-default": RepresentativeCase(
        input_path=Path("tests/fixtures/pdb/1aho.pdb"),
        summary=StructureSummary(
            chain_ids=("A",),
            chain_summaries=(ChainSummary("A", 64, 962),),
            residue_count=64,
            ligand_count=0,
            atom_count=962,
            hydrogen_atom_count=457,
            component_counts=tuple(
                sorted(
                    {
                        "ALA": 3,
                        "ARG": 3,
                        "ASN": 4,
                        "ASP": 4,
                        "CYS": 8,
                        "GLN": 1,
                        "GLU": 3,
                        "GLY": 7,
                        "HIS": 2,
                        "ILE": 1,
                        "LEU": 2,
                        "LYS": 5,
                        "PHE": 1,
                        "PRO": 3,
                        "SER": 2,
                        "THR": 3,
                        "TRP": 1,
                        "TYR": 7,
                        "VAL": 4,
                    }.items()
                )
            ),
            first_residue="ATOM:VAL:A:1",
            last_residue="ATOM:HIS:A:64",
            semantic_digest="4bb2c51f81a9321208731b67235fd43e37f896cce4427c35796b076378576d26",
        ),
    ),
    "1cjc-heavy-keep-ligand": RepresentativeCase(
        input_path=Path("tests/fixtures/corpus/pdb1cjc.ent"),
        summary=StructureSummary(
            chain_ids=("A",),
            chain_summaries=(ChainSummary("A", 455, 3505),),
            residue_count=455,
            ligand_count=1,
            atom_count=3558,
            hydrogen_atom_count=0,
            component_counts=tuple(
                sorted(
                    {
                        "ALA": 39,
                        "ARG": 37,
                        "ASN": 7,
                        "ASP": 27,
                        "CYS": 5,
                        "FAD": 1,
                        "GLN": 18,
                        "GLU": 27,
                        "GLY": 42,
                        "HIS": 10,
                        "ILE": 18,
                        "LEU": 50,
                        "LYS": 17,
                        "MET": 6,
                        "PHE": 15,
                        "PRO": 35,
                        "SER": 24,
                        "THR": 24,
                        "TRP": 6,
                        "TYR": 8,
                        "VAL": 40,
                    }.items()
                )
            ),
            first_residue="ATOM:THR:A:6",
            last_residue="ATOM:HIS:A:460",
            semantic_digest="8ea7ec6d8cd75dd5c01310dd598b724395e0667cb157e2c74272bbb864f943f1",
        ),
    ),
    "1aar-heavy-chain-1": RepresentativeCase(
        input_path=Path("tests/fixtures/corpus/pdb1aar.ent"),
        summary=StructureSummary(
            chain_ids=("A",),
            chain_summaries=(ChainSummary("A", 76, 602),),
            residue_count=76,
            ligand_count=0,
            atom_count=602,
            hydrogen_atom_count=0,
            component_counts=tuple(
                sorted(
                    {
                        "ALA": 2,
                        "ARG": 4,
                        "ASN": 2,
                        "ASP": 5,
                        "GLN": 6,
                        "GLU": 6,
                        "GLY": 6,
                        "HIS": 1,
                        "ILE": 7,
                        "LEU": 9,
                        "LYS": 7,
                        "MET": 1,
                        "PHE": 2,
                        "PRO": 3,
                        "SER": 3,
                        "THR": 7,
                        "TYR": 1,
                        "VAL": 4,
                    }.items()
                )
            ),
            first_residue="ATOM:MET:A:1",
            last_residue="ATOM:GLY:A:76",
            semantic_digest="de2352b0a28b8a44e119a7f5c4110daede9741b14e03029ab27d08d7506bd560",
        ),
    ),
    "1cjc-hydrogen-keep-ligand": RepresentativeCase(
        input_path=Path("tests/fixtures/corpus/pdb1cjc.ent"),
        summary=StructureSummary(
            chain_ids=("A",),
            chain_summaries=(ChainSummary("A", 455, 7043),),
            residue_count=455,
            ligand_count=1,
            atom_count=7096,
            hydrogen_atom_count=3538,
            component_counts=tuple(
                sorted(
                    {
                        "ALA": 39,
                        "ARG": 37,
                        "ASN": 7,
                        "ASP": 27,
                        "CYS": 5,
                        "FAD": 1,
                        "GLN": 18,
                        "GLU": 27,
                        "GLY": 42,
                        "HIS": 10,
                        "ILE": 18,
                        "LEU": 50,
                        "LYS": 17,
                        "MET": 6,
                        "PHE": 15,
                        "PRO": 35,
                        "SER": 24,
                        "THR": 24,
                        "TRP": 6,
                        "TYR": 8,
                        "VAL": 40,
                    }.items()
                )
            ),
            first_residue="ATOM:THR:A:6",
            last_residue="ATOM:HIS:A:460",
            semantic_digest="7e79b110444265a84d2c3eed65046b500281c00f20f7d630508fda4ec0053858",
        ),
    ),
    "1afc-hydrogen-his-protonated": RepresentativeCase(
        input_path=Path("tests/fixtures/corpus/pdb1afc.ent"),
        summary=StructureSummary(
            chain_ids=("A", "B", "C", "D", "E", "F", "G", "H"),
            chain_summaries=(
                ChainSummary("A", 127, 2031),
                ChainSummary("B", 127, 2031),
                ChainSummary("C", 127, 2031),
                ChainSummary("D", 127, 2031),
                ChainSummary("E", 127, 2031),
                ChainSummary("F", 127, 2031),
                ChainSummary("G", 127, 2031),
                ChainSummary("H", 127, 2031),
            ),
            residue_count=1016,
            ligand_count=0,
            atom_count=16248,
            hydrogen_atom_count=8104,
            component_counts=tuple(
                sorted(
                    {
                        "ALA": 40,
                        "ARG": 40,
                        "ASN": 40,
                        "ASP": 48,
                        "CYS": 16,
                        "GLN": 48,
                        "GLU": 72,
                        "GLY": 96,
                        "HIS": 40,
                        "ILE": 48,
                        "LEU": 136,
                        "LYS": 88,
                        "MET": 8,
                        "PHE": 48,
                        "PRO": 48,
                        "SER": 56,
                        "THR": 64,
                        "TRP": 8,
                        "TYR": 48,
                        "VAL": 24,
                    }.items()
                )
            ),
            first_residue="ATOM:LYS:A:10",
            last_residue="ATOM:PRO:H:136",
            semantic_digest="90a351e83f57411da43af739bfb3b7b38c14579130978fc537e1c835ce3cf879",
        ),
        protonate_histidines=True,
    ),
}
