# Methods

 Three methods implemented in **CICOBase** package: `:CICO_ONE_PASS`,
 `:LIN_EXTRAPOL`,
 `:QUADR_EXTRAPOL`.

The main function for CI endpoints estimation is `get_interval` (see API section). Estimation method is defined by the keyword argument `method`. It supports one of the above values. Default is `:CICO_ONE_PASS`.

## :CICO\_ONE\_PASS

The method uses the one-pass calculation of confidence interval endpoint and  utilizes **Inequality-based Constrained Optimization**
for efficient determination of confidence intervals and detection of “non-identifiable” parameters.

 The method internally calls [`NLopt`](https://nlopt.readthedocs.io/en/latest/) algorithm to build an **augmented objective function** with [`LN_AUGLAG`](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#augmented-lagrangian-algorithm) algorithm. Optimization algorithm choice is described in Optimizers section.

When an endpoint has status `:SCAN_BOUND_REACHED`, `CICO_ONE_PASS` returns
`endpoint.value = nothing` and an empty `endpoint.profilePoints`.

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
