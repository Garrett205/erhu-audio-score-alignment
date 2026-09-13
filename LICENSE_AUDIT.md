# License audit

Scope: selected paper-authored source, its local source ancestors, and installed/
archived third-party dependency notices. No original root LICENSE was located.
MIT is applied to this independently extracted author code under the task's
authorization, using a project contributor attribution because legal author
metadata was not provided. This scan is provenance evidence, not independent
verification of every contributor's ownership.

| Component | Local evidence | Treatment |
|---|---|---|
| Proposed pipeline, DTW cores, wrappers, metrics | Project-authored frozen files; no third-party copyright headers located in selected source | MIT snapshot |
| TAPE | Archived LICENSE: GNU AGPL version 3 | Source and weights excluded; separate external dependency; original license notice retained |
| STRAdi | Checkout LICENSE: Apache 2.0 | Source and weights excluded; external dependency; original license notice retained |
| Parangonar | Installed 3.3.3 metadata: Apache-2.0 | Source/weights excluded; install upstream package |
| torchcrepe, df0-pitch, librosa and numerical libraries | Installed dependencies, not vendored | Remain under their original licenses; MIT does not relicense them |

The TAPE AGPL license must not be replaced by this repository's MIT license.
The MIT license covers this repository's authored files only; third-party license
texts in `docs/third_party/` retain their original terms. No permissive relicense
of TAPE, model weights, CCOM-HuQin or recordings is claimed. Use the original
projects' installation and license instructions when obtaining dependencies.
