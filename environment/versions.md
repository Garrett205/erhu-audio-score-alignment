# Frozen environment evidence

Source: the 10 September 2026 software freeze archived with the V17 supplement.
Personal installation paths and the original remote are intentionally omitted.

| Component | Recorded version |
|---|---|
| Python | 3.10.19 |
| torchcrepe | 0.0.24 (not the separate `crepe` package) |
| dSWIPE | df0-pitch 1.0.1; df0 module 0.1.1 |
| NumPy | 2.2.6 |
| SciPy | imported 1.15.2; Conda list 1.15.3 |
| SoundFile | imported 0.14.0; distribution metadata 0.13.1; Conda PySoundFile 0.14.0 |
| librosa | 0.11.0 |
| Matplotlib | 3.10.8 |
| pandas | 2.3.3 |
| PyTorch | 2.6.0+cu124 |
| music21 | 9.9.1 |
| openpyxl | 3.1.5 |
| tqdm | 4.67.3 |

dSWIPE entry point: `dswipe=df0.main:dswipe`. It had no working `--version`;
the historical Windows launcher SHA-256 was
`a5b1c07b50d65bd68e2384f4277a8821c89d4ea79de3c39948c473659766da65`.
This launcher hash is platform-specific, not a portable dependency requirement.

No environment was upgraded, repaired or installed for this extraction.
requirements.txt records the imported SciPy version and SoundFile distribution
metadata; installing it cannot be claimed to recreate the mixed historical
SoundFile state. The discrepancy remains unresolved. A clean-environment
installation and full inference reproduction have not been certified.

Optional frozen baseline packages: Parangonar 3.3.3, miditok 3.0.6.post1,
symusic 0.6.0, dtaidistance 2.4.0; obtain their own dependency environments.
TAPE and STRAdi model identities are in `baselines/README.md`.
