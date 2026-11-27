# Methods

 Five methods implemented in **CICOBase** package: `:CICO_ONE_PASS`,
 `:DIRECT_SLSQP`,
 `:DIRECT_COBYLA`,
 `:LIN_EXTRAPOL`,
 `:QUADR_EXTRAPOL`.

The main function for CI endpoints estimation is `get_interval` (see API section). Estimation method is defined by the keyword argument `method`. It supports one of the above values. Default is `:CICO_ONE_PASS`.

## :CICO\_ONE\_PASS

The method uses the one-pass calculation of confidence interval endpoint and  utilizes **Inequality-based Constrained Optimization**
for efficient determination of confidence intervals and detection of “non-identifiable” parameters.

 The method internally calls [`NLopt`](https://nlopt.readthedocs.io/en/latest/) algorithm to build an **augmented objective function** with [`LN_AUGLAG`](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#augmented-lagrangian-algorithm) algorithm. Optimization algorithm choice is described in Optimizers section.

`get_endpoint` and `get_interval` accept `penalty_tol` for the local solver's
penalized objective, defaulting to `1e-6`. The relative function tolerance
of the local solver is fixed at `0.0`. `penalty_tol` controls changes in the full penalized objective; it
does not guarantee the accuracy of the profile loss. The outer solver uses
`scan_tol` in scan coordinates, independently of `penalty_tol`. `loss_tol`
sets the allowed positive loss residual for AUGLAG's feasibility check and
defaults to `1e-3` in loss units.
For `:CICO_ONE_PASS`, `:DIRECT_SLSQP` and `:DIRECT_COBYLA`, `loss_tol=0.0` uses zero
constraint tolerance. The tolerance applies to NLopt's stopping checks and
does not establish that the returned point is close to the critical loss.

For local NLopt algorithms that use `tolg`, including the TNEWTON family,
the gradient stopping tolerance is set internally to `1e-2` and applies to
the penalized local objective.

When an endpoint has status `:SCAN_BOUND_REACHED`, `CICO_ONE_PASS` returns
`endpoint.value = nothing` and an empty `endpoint.profilePoints`.

## :DIRECT_SLSQP

`:DIRECT_SLSQP` applies NLopt's `:LD_SLSQP` directly to the scan objective
and the loss inequality. Select it using `get_interval(theta_init, theta_num,
loss_func, :DIRECT_SLSQP; loss_crit, loss_grad=:AUTODIFF)`. The method also
supports a scan function when `scan_grad` is supplied. It requires
`loss_grad` as `:AUTODIFF`, `:FINITE` or a function.

`scan_tol` controls the absolute objective tolerance, `loss_tol`
controls NLopt's feasibility tolerance (default `1e-3`), and `max_iter` limits objective
evaluations. `penalty_tol`,
`local_alg` and `local_initial_step` do not configure this direct solver.
`:ROUNDOFF_LIMITED` and other unclassified NLopt results produce
`:UNKNOWN_STOP`. The experimental comparison under
`hypo/02-nlopt-auglag` found a number of unvalidated endpoints on Bruno.

## :DIRECT_COBYLA

`:DIRECT_COBYLA` applies NLopt's derivative-free `:LN_COBYLA` directly to the
scan objective and loss inequality. It accepts `scan_tol`, `loss_tol`, `max_iter`
and `local_initial_step`, without requiring gradients. `loss_tol` defaults to
`1e-3` and controls NLopt's constraint feasibility tolerance. When supplied,
`local_initial_step` uses optimization coordinates.

## Checking a parameter interval

`validate_interval` checks the two sides of a parameter interval independently.
The absolute tolerances are specified in loss units:

```julia
check = validate_interval(result;
    boundary_loss_atol = 1e-3,
    profile_loss_atol = 1e-3)
check.left.outcome, check.right.outcome
```

For `:BORDER_FOUND_BY_SCAN_TOL`, it reevaluates the loss and restarts profile
minimization with the parameter fixed. `:SCAN_BOUND_REACHED` and
`:BORDER_FOUND_BY_LOSS_TOL` are not checked and have outcome `:not_checked`.
`:pass` means the checks passed; `:fail` reports a failed check;
`:inconclusive` means a check could not be completed. The original endpoint
and status are not modified. Local
optimization cannot establish a global optimum. Intervals of a general
`scan_func` are not supported. See [`validate_interval`](@ref) for settings
and report details; the tolerances above are examples, not universal values.

## :LIN_EXTRAPOL

The method examines profile likelihood function by making steps in both directions from the optima. 
Each step is derived as a linear extrapolation: `y = a*x + b`.

## :QUADR_EXTRAPOL

The method examines profile likelihood function by making steps in both directions from the optima. 
Each step is derived as a quadratic extrapolation: `y = x^2*a + x*b + c`.
