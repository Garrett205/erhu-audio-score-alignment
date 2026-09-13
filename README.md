# Erhu Audio-to-Score Alignment

Code accompanying the manuscript "Reliability-Aware Audio-to-Score Event Alignment for Erhu Performance Analysis."

## Scope

This repository contains only the pitch-front-end, event-candidate construction,
and score-alignment components reported in the paper.
It does not contain the broader Erhu performance-assessment project.

The coupled scripts remain together in `src/pipeline`: `pitch.py` is the frontend,
`onset.py` constructs candidates, `mxml.py` represents score events, `os.py`
normalizes the performance axis, and `dp1.py`–`dp3.py` call the actual frozen
`*_legacy.py` implementations. Those filenames denote active dependencies.
The retained stage-local error/level columns are historical alignment diagnostics;
no downstream proficiency model or assessment system is included.

## Pipeline

Audio → CREPE+dSWIPE reliability-aware frontend → performance-boundary gate
→ event candidates → global normalization → DP1 → DP2 → DP3
→ unmatched-only Mel rescue → event correspondence.

The start/end gate is **candidate-domain performance-boundary gating**, not waveform trimming.
Its frozen thresholds are 1800 Hz and periodicity 0.01; start uses 40/40 frames,
end uses 30/−28 frames. The original strict comparisons and scan order are retained.

## Installation

Use Python 3.10.19. See [environment evidence](environment/versions.md) before installing:

```sh
conda env create -f environment/environment.yml
conda activate erhu-paper
python -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r environment/requirements.txt
```

The mixed historical environment is not a verified clean-install lockfile.
Third-party baseline code and weights must be obtained from their original repositories;
see [baseline setup](baselines/README.md) and [license audit](LICENSE_AUDIT.md).

## Data

CCOM-HuQin: [Zhang et al. (2023), dataset article](https://doi.org/10.5334/tismir.146).
Obtain data independently and follow its original terms. No audio, scores,
annotations, consent forms, or participant-level records are distributed here.
Private score-following recordings are not publicly distributed due to
participant-consent/privacy restrictions.

## Reproduction

```sh
python run_pipeline.py --input inputs/performance.wav --score inputs/score.musicxml --output work/example
python verify_snapshot.py
```

Use a new output directory. The launcher copies supplied inputs there and invokes
the unchanged stage logic. Historical plots/logs can be generated in this work
directory and are ignored by Git. Final correspondence is in `work/example/final/`.
Input/output paths are arguments or environment variables, never a workstation path.

Public CCOM data permit frontend and pYIN/TAPE pitch evaluation after users obtain
the matching inputs and references. The exact downloaded release identifier was
not retained; a fresh download alone does not guarantee the exact paper population.
Private alignment results cannot be completely rerun from this repository.
Code and parameters are supplied, but the private audio and full human labels are absent.
Not all paper numbers are reproducible from publicly available data.
See [reproduction details](docs/REPRODUCIBILITY.md) and [method map](docs/METHOD_TO_CODE_MAP.md).

## Citation

Publication metadata is pending. Cite the manuscript title above until final
author, year, DOI and venue metadata are available. `CITATION.cff` is a documented draft.
