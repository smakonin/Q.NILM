# Reproducing the research release

The release contains code, tests, protocols and bounded experimental evidence.
It does not distribute manuscript files, journal correspondence, publication
figures, or the D-Wave proposal. Original household datasets and licensed
third-party papers are not included.

## Installation and evidence

1. Extract `QNILM-code.zip` into an empty directory, or clone the corresponding
   release tag. Create a fresh Python environment and install
   `python -m pip install -e '.[dwave,ibm,stage-b,stage-e]'`.
   `requirements-verified.txt` records the versions used for release checks.
2. Extract all `QNILM-evidence-*.zip` files into the same directory to restore
   `results/`. The release manifest lists every included file and SHA-256.
   The archives are experiment evidence, not additional independent trials.
3. Run `PYTHONPATH=src:. python -m unittest discover -s tests -q`.
   The small archived fixtures required by the suite are also in the code
   repository. Large immutable records live in the evidence assets.
4. Run offline audits with fresh output paths, for example:

   ```sh
   PYTHONPATH=src:. python scripts/audit_quantum_heldout_hardware.py --require-complete --output full-ibm-check.json
   PYTHONPATH=src:. python scripts/audit_physical_diagnostics.py --folder results/quantum_diagnostics/physical_001 --output physical-check.json
   PYTHONPATH=src:. python scripts/audit_quantum_diagnostic_counts.py --folder results/quantum_diagnostics/ladder_date_repeat_001 --output repeat-counts-check.json
   PYTHONPATH=src:. python scripts/audit_quantum_diagnostic_ladder.py --folder results/quantum_diagnostics/ladder_date_repeat_001 --output repeat-ideal-check.json
   ```

The release excludes the REFIT report's generated LaTeX table, an author-only
publication artifact. Its numerical inputs and report generator are retained;
recreate it in a separate output directory if required. No scientific result
record is altered to hide an unsuccessful experiment.

## Source data and execution limits

For full raw-data rereads obtain R1Hz (DOI `10.7910/DVN/RCB5VJ`) and the REFIT
cleaned data (DOI `10.15129/9ab14b0e-19ac-4279-938f-27f643078cec`) from their
original repositories, under their terms. Archived plans preserve original
source paths as provenance. Explicitly record any relocation; changing a
plan must not be represented as retaining its original freeze hash.

Offline checks do not require IBM credentials. Do not run hardware submission
commands merely to inspect results. Hardware execution requires the user's
own authorized account and budget; free-plan guards and no-duplicate-job
checks are part of the runners, not a guarantee of future provider terms.

The completed IBM accuracy campaign remains one full run. The September 18
24-circuit repeat is a component-feasibility diagnostic on a second date,
not a new full accuracy evaluation. Protocols document validation/test
exposure, fallback rules and limitations. No quantum advantage is claimed.

## Distribution integrity

`scripts/check_distribution_boundary.py` checks all reachable Git commits for
prohibited publication/credential paths. It is also run in CI. The release
asset inventory and credential scan are additional checks, not a claim that
all possible secrets can be detected automatically. Never merge histories
from unrelated or private publication checkouts into this repository.
