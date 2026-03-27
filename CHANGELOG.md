# Changelog

## 0.1.0

Initial public release of the rewritten `pras` package.

- add a typed `src/pras` package with a small public API centered on
  `process_structure()` and `ProcessOptions`
- support canonical PDB/mmCIF ingress through `gemmi`
- implement missing heavy-atom repair and hydrogen placement on the canonical
  structure model
- package the vendored `FASPR` executable as the first side-chain packing
  backend specialization
- add direct unit regressions, repo-local repair fixtures, typed package
  metadata, and release-ready licensing notices
