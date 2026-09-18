# Release verification — 18 September 2026

This is a code/results research snapshot, not a claim of journal acceptance
or quantum advantage. Public availability is scheduled immediately before
submission; the repository remains private during preparation.

Checks performed on the separately assembled code-only publishing checkout:

- All 439 unittest cases passed without skips (Python 3.12.14, macOS ARM64).
- Small synthetic, ideal IBM-simulator and local Ocean SA/tabu smoke runs
  completed. These smoke outputs are software checks, not replacement
  scientific experiments or hardware results.
- A fresh independent audit passed for all 129 full IBM result records:
  2,451 circuits and 627,456 raw shots, with accounted usage and fallback
  reconciliation. This is not a fresh reread of original household data.
- A fresh physical-control audit passed for 228 circuits and 94,208 shots.
  Bootstrap interval quantiles were checked, not every bootstrap fit rerun.
- The September 18 component repeat passed independent raw-count and
  compiled-ideal audits: 24 circuits, 24,576 shots and 9 seconds finalized
  usage. It is a second component date, not a second full NILM campaign.
- New static/development summaries were independently reconciled from
  per-window errors for 10 reporting rows and 30 background candidates.
- The selected distribution files were scanned for exact saved IBM-token
  matches and high-confidence credential signatures, including decoded gzip
  and NPZ payloads. No matches were found. This is a bounded scan, not a
  guarantee that every possible secret format is detectable.

The release manifest binds distributed bytes and revision identifiers.
Historical run plans, audit receipts and failures remain immutable. Original
data, credentials and publication files are excluded. The code repository
has a new root history, with no imported private publication commits.

See `requirements-verified.txt`, the experiment protocols and
[reproduction instructions](reproducing_release.md). Local checks are not
represented as tests on every operating system or as a historical rerun of
every archived experiment.
