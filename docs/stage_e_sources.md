# Stage E: published-baseline source and feasibility audit

Checked 2026-09-14. This is a source audit, not an experimental result.
It does not convert a generic mixed-integer solver into a reproduction of a
published NILM method.

## Balletti, Piccialli and Sudoso (2022)

Verified citation: Marco Balletti, Veronica Piccialli, and Antonio M. Sudoso,
“Mixed-Integer Nonlinear Programming for State-Based Non-Intrusive Load
Monitoring,” *IEEE Transactions on Smart Grid*, 13(4), 3301–3314, 2022,
[DOI 10.1109/TSG.2022.3152147](https://doi.org/10.1109/TSG.2022.3152147).
The [author-deposited accepted manuscript](https://iris.uniroma1.it/retrieve/e383532e-89b2-15e8-e053-a505fe0a3de9/Balletti_postprint_Mixed-integer_2022.pdf.pdf)
and [arXiv record](https://arxiv.org/abs/2106.09158) are publicly accessible.

The full method requires alternating autoregressive clustering, parameter
estimation, constrained binary-quadratic disaggregation, and autoregressive
postprocessing. Its penalties encode temporal smoothness and time-dependent
device sparsity. Appliance-specific constraints include mutually exclusive
nonzero states, always-on devices, permitted transitions, minimum/maximum
dwell times, activation counts, energy budgets, and predicted load below mains.
Training supplies the state/usage priors; validation selects two regularization
parameters. The experiments use 60-second measurements and AR order three.
See Sections II–IV, especially equations (2)–(22) and Algorithm 1. Simply
retaining Q.NILM's learned states and replacing its optimizer does not implement
these modelling stages.

### Official implementation

The paper links [antoniosudoso/nilm-bqp](https://github.com/antoniosudoso/nilm-bqp).
Inspected revision: `1d8c3addfd4cd08ac75ea4aea47ac8e1a04ac200`.
It has a [GPL-3.0 license](https://github.com/antoniosudoso/nilm-bqp/blob/1d8c3addfd4cd08ac75ea4aea47ac8e1a04ac200/LICENSE),
AMPL models, Gurobi run scripts, Python parameter-analysis helpers, and
dataset-specific parameter/data examples for AMPDS, UK-DALE, and REFIT.

The [REFIT model](https://github.com/antoniosudoso/nilm-bqp/blob/1d8c3addfd4cd08ac75ea4aea47ac8e1a04ac200/code/bqp_formulation/REFIT/3/nilm_binary.mod)
contains two conventions requiring an explicit reproduction choice: its
squared residual has coefficient 0.5, and its sparsity term starts at the
second sample. The manuscript equations show an unhalved residual and a
sparsity term beginning at the first sample. The
[run script](https://github.com/antoniosudoso/nilm-bqp/blob/1d8c3addfd4cd08ac75ea4aea47ac8e1a04ac200/code/bqp_formulation/REFIT/3/nilm_binary.run)
uses Gurobi with presolve enabled; it reads precomputed appliance parameters
and clipping sets. A separate
[postprocessing script](https://github.com/antoniosudoso/nilm-bqp/blob/1d8c3addfd4cd08ac75ea4aea47ac8e1a04ac200/code/bqp_formulation/REFIT_PP/3/ar_adjustment.run)
produces recursive, nonnegative AR-adjusted predictions. Fixed example
parameters must not be silently substituted for learning on our own split.

### Feasibility in this workspace

AMPL and Gurobi executables were not found on the current shell path; this
does not establish whether the user has a license elsewhere. Neither was
installed or licensed by this audit. Reusing upstream source requires
preserving its license and attribution, not relabelling it as Q.NILM code.
An independently written formulation is possible, but requires explicit
parameter-learning rules, appliance metadata, numerical equivalence tests,
training/validation selection, and frozen evaluation before it can be called
a faithful published-baseline comparison. A solver-only experiment should
instead be labelled a matched-objective classical control.

## Li, Zhai and Zhou (2025)

Verified citation: Haozhe Li, Qiaozhu Zhai, and **Yuzhou Zhou**,
“Non-Intrusive Load Monitoring via Binary Integer Search and Transient Load
Feature Integration,” *IEEE Transactions on Smart Grid*, 16(4), 3408–3418,
2025, [DOI 10.1109/TSG.2025.3564472](https://doi.org/10.1109/TSG.2025.3564472).
Full author names are confirmed by
[IEEE-deposited Crossref metadata](https://api.crossref.org/works/10.1109/TSG.2025.3564472).

The publisher abstract describes two stages: binary integer search produces
a feasible candidate space, and transient load features select the final
solution. It states that a few appliance electrical parameters replace
device-level training traces. This is not equivalent to generic binary
enumeration, branch-and-bound, or solving Q.NILM's existing objective.
[Publisher abstract](https://doi.org/10.1109/TSG.2025.3564472).

The initial public search did not retrieve full methods or author code. The
user subsequently supplied the full eleven-page IEEE paper, which was read
in full and whose equations, algorithm and experimental-input pages were
visually checked. File SHA-256:
`382f992ddd2cdaae51461ffb70e9d121461fcc75c393313f8d8f94e240b40046`.
The licensed PDF remains outside the repository; it is not redistributed.

The full text specifies experiments on **60 Hz active power**, using EMBED
and BLUED (Sections I and IV-A). Its event detector isolates transient spans;
DTW, LFIG-DTW or FLF-DTW compares their shapes with an appliance-labelled
event-example dictionary. Clustering supplies medoids, and appliance names
require user confirmation (Section III-B). The implementation additionally
depends on the cited long-transient detector, clustering/granulation methods,
appliance operating ranges and explicit threshold choices.

Equations (3)--(8) specify a mode-constrained feasible binary state set,
10% trimming at both ends of each steady window, optional partial-load
Fast-BISA and historical-data compression thresholds. Equation (9) combines
normalized aggregate-step mismatch, transient distance and switch count in
a shortest-path graph. Algorithm 1 gives binary-search pseudocode, but this
component alone is not the full disaggregation method. A faithful executable
also needs explicit handling of zero denominators and empty candidate sets;
these must not be silently invented as if supplied author settings.

**User scope decision, 2026-09-14:** keep the current low-frequency study and
explicitly defer the full Li comparison. R1Hz supplies 1 Hz measurements and
the current benchmark scores 30-second blocks; REFIT is lower-rate still.
Neither contains the original 60 Hz transient waveforms. Interpolation
cannot recover those missing measurements. No EMBED/BLUED download or new
high-rate experiment is authorized or run. Li remains a related-work source;
any future low-rate BISA-only control must be labelled an adaptation/ablation,
not a reproduction of the reported full method.

## Stage E reporting decision

Keep separate evidence gates:

1. **Matched-objective solver comparison:** independently implemented MIP,
   exact DP/enumeration, and stochastic solvers with the same objective and
   inference inputs. This can be completed using available open-source tools.
2. **Published low-frequency NILM algorithm comparison:** the full Balletti
   method on a declared common protocol, with all departures documented.
   This is not completed by gate 1. The user has explicitly deferred Li's
   full transient-feature method beyond the current low-frequency scope.

The previously inspected test periods cannot become a new blind confirmatory
test by changing the solver or the baseline. Preserve the original archives,
freeze any new protocol before execution, train/tune only in the assigned
training/validation periods, and report follow-on comparisons as post-exposure.
The full original Stage E programme remains partially complete: gate 1 does
not satisfy gate 2. For the **18 September 2026 submission scope**, gate 1 is
the completed matched-objective benchmark and gate 2 is explicitly excluded.
The manuscript now disclaims state-of-the-art NILM superiority, rather than
promising an unperformed Balletti comparison. This is a scope decision, not
new experimental evidence or a claim that the published method was reproduced.
