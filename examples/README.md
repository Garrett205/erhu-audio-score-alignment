# Bring your own authorized inputs

No example audio, MusicXML, annotations, participant manifests or model weights
are bundled. Obtain CCOM-HuQin under the original terms:
Zhang Y., Zhou Z., Li X., Yu F., Sun M. (2023), “CCOM-HuQin: An annotated
multimodal Chinese fiddle performance dataset”, TISMIR 6(1), 60–74.
https://doi.org/10.5334/tismir.146

Private score-following recordings: not publicly distributed due to
participant-consent/privacy restrictions.

For your own audio/score pair use `run_pipeline.py --input ... --score ... --output ...`.
The output must be a new directory; original supplied inputs remain untouched.
