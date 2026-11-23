# Choosing a local optimizer

`CICO_ONE_PASS` uses NLopt's augmented Lagrangian algorithm (`LN_AUGLAG`)
with the local optimizer selected by `local_alg`. The practical choices below
are based on the direct-scale numerical experiments, not guarantees for all
problems.

The experiments compared two ways of handling parameter bounds: **method 1**,
used by the current implementation, passes them as NLopt box bounds; **method 2**
adds inequality constraints. Method 2 can evaluate intermediate points outside
the parameter bounds. Use particular care if the objective is undefined there.

The reported results use `local_initial_step=0.5`. The five configurations were
unbounded parameters, and bounds of ±100 and ±1e6 with each of the two methods.
Each optimizer and configuration had 58 endpoint sides with known reference
status and value; another 26 sides had no reference. A match requires the expected
status and an endpoint within `1e-4`. Bounds on the scanned parameter remained
±9. See `draft/initial-step-0.5/summary.csv` and `protocol.md` in the repository
for the full results and settings. These tests do not establish correctness for
all objectives, parameter scales, or active parameter bounds.

## Recommended candidates

### Derivative-free

- **`LN_NELDERMEAD` — first choice.** Matched 58/58 in every configuration.
  Set `local_initial_step` explicitly: with the default step, the wide-bound
  runs fell to 53/58 with method 1 and 56/58 with method 2. Both methods matched
  all references with step 0.5; the four recommended optimizers had identical
  reference results with both bound methods. The step is in optimization coordinates, so 0.5 is an
  experimental setting rather than a universal choice.
- **`LN_SBPLX` — alternative to check independently.** Matched 57/58 in every
  configuration with step 0.5. It uses the supplied start and initial step;
  setting the step improved the bounded runs. Neither bound method was superior
  by the number of matches at this step. Investigate discrepancies rather than
  assuming its endpoint is correct.

### Gradient-based

These algorithms require a loss gradient; `loss_grad=:AUTODIFF` uses ForwardDiff.
They ignore `local_initial_step`, but use the supplied starting point.

- **`LD_TNEWTON` — first gradient-based candidate.** Matched 58/58 with both
  bound methods and both widths, and 57/58 without finite parameter bounds.
  Neither bound method had more matches in these tests.
- **`LD_LBFGS` — bounded-problem alternative.** Matched 58/58 with both bound
  methods and both widths, but 53/58 without finite parameter bounds. Neither
  bound method had more matches in the bounded tests.

## Methods to avoid as defaults

- **`LN_NEWUOA`** does not enforce NLopt box bounds. It matched only 50/58 in
  each configuration. `LN_NEWUOA_BOUND` is a separate algorithm and was not
  included in these experiments.
- **`LN_BOBYQA`** matched 49/58 in each configuration. With box bounds it can
  shift the starting point when it is within one initial step of a bound; this
  conflicts with a requirement to start exactly at the supplied point. This
  behavior has not been established as the cause of the observed mismatches.
- **`LN_PRAXIS`** was sensitive to bound handling: with step 0.5, it matched
  33/58 and 51/58 with method 1 for wide and narrow bounds, versus 55/58 for
  both with method 2. Five of its endpoints passed the validator despite
  differing from the known reference. It uses the initial step; for a vector
  of steps, NLopt reduces it to the smallest absolute component.
- **`LN_COBYLA`** matched 49–52/58 with step 0.5; several runs reached the
  iteration limit. It uses the supplied initial step. With no explicit step,
  some intervals timed out in the unbounded and method 2 runs.
- **`LD_MMA` and `LD_CCSAQ`** matched 52–55/58 depending on the configuration,
  including iteration-limit stops and incorrect reported boundaries. Both use
  the initial step as an initial search scale; setting it to 0.5 did not make
  them reliable on this test set.
- **`LD_SLSQP`, `LD_VAR1`, `LD_VAR2`, and the other `LD_TNEWTON` variants**
  were less consistent across bounded and unbounded configurations. For example,
  `LD_SLSQP` matched 58/58 with method 2 and either finite width, but 47/58
  without finite bounds. These methods ignore `local_initial_step`.

For any optimizer, check the returned status and consider `validate_interval`
for endpoints reported as `:BORDER_FOUND_BY_SCAN_TOL`. The validator is a local
check and can accept an endpoint that differs from the known answer; it does
not prove global correctness. `:SCAN_BOUND_REACHED` is not checked by it.
