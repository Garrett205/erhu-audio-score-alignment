# Snapshot validation

- 30 copied/extracted source files: source SHA-256 verified unchanged after extraction.
- Pitch, onset, DP1/DP2/DP3 core files and helpers preserve original algorithms.
- Eight synthetic score constructions: every retained note-table and legacy-table
  column equals the original parser exactly (including score IDs/order, tie,
  grace and chord handling). Removed columns are downstream technique exports.
- Synthetic DP1 → DP2 → DP3 stage execution passed; final ordered IDs preserved.
- Synthetic gate edge cases, hard-invalid pitch handling, short octave repair,
  simple ordered DP1 correspondence and original baseline readout checks passed.
- Portable pipeline `--help` passed. Source compiles without producing bytecode.
- No private inputs, real paper alignment reruns, model inference, fresh
  environment installation or new human scoring were used for these checks.

Source hashes and definition comparisons are in `SOURCE_PROVENANCE.json`;
new adapter provenance is separate in `ADAPTER_PROVENANCE.json`.
The new generic launchers have not been validated on the private twelve-piece
benchmark; synthetic validation does not establish full numerical reproduction.
`CITATION.cff` remains a draft pending verified author/publication metadata.
