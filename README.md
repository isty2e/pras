# PRAS

`pras` is a typed Python package for protein repair based on the original
[PRAS](https://github.com/osita-sunday-nnyigide/Pras_Server) project.

Current implemented scope:

- PDB and mmCIF ingress through `gemmi`
- missing heavy-atom repair
- hydrogen placement
- canonical workflow entrypoint via `process_structure()`
- optional packaged FASPR backend for side-chain packing guidance

Current deferred scope:

- analysis execution (`secondary structure`, `Ramachandran`)
- broader nonstandard-residue support

## Installation

Install the package from the repository root:

```bash
pip install .
```

This build also packages the vendored FASPR executable and rotamer library used
by the optional side-chain packing backend.

## Usage

```python
from pathlib import Path

from pras import ProcessOptions, process_structure
from pras.model import HydrogenPolicy, LigandPolicy

result = process_structure(
    Path("tests/fixtures/pdb/1aho.pdb"),
    options=ProcessOptions(
        hydrogen_policy=HydrogenPolicy.ADD_MISSING,
        ligand_policy=LigandPolicy.KEEP,
    ),
)

structure = result.structure
if result.has_errors():
    raise RuntimeError(result.issues)
```

If you want to write the repaired structure back out:

```python
from pathlib import Path

from pras.io import write_structure

write_structure(structure, Path("output.pdb"))
```

## Development

Run the permanent verification surface with:

```bash
ruff check src/pras tests --extend-select=I,UP --fix
basedpyright src/pras tests
pytest tests/unit -q
```

## License

This repository is licensed under [MIT](LICENSE).
Third-party provenance is summarized in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
