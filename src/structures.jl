"""
    struct ProfilePoint
        value::Float64               # x value of profile point
        loss::Float64                # y value of profile point (loss function value)
        params::Array{Float64, 1}    # vector of optimal values of `loss_func` arguments
        ret::Symbol                  # `NLopt.optimize()` result code
        counter::Union{Int, Nothing} # number of `loss_func` evaluations
        force_stop_code::Int         # reason code for `:FORCED_STOP` (0 otherwise)
    end

Structure storing one point from a profile function. `ret` is the result code
returned by NLopt, or `:OPTIMIZATION_SKIPPED` when no optimization was run.
Common NLopt codes are:

- `:FTOL_REACHED`: the optimizer met its objective-value tolerance; this alone
  does not establish that the loss reached its critical value.
- `:MAXEVAL_REACHED`: the optimizer reached its evaluation limit.
- `:FORCED_STOP`: the optimizer was stopped externally. `force_stop_code`
  distinguishes known callback failures from other forced stops.

Other NLopt result codes are preserved in `ret`.

"""
struct ProfilePoint
    value::Float64
    loss::Float64
    params::Array{Float64, 1}
    ret::Symbol
    counter::Union{Int, Nothing}
    force_stop_code::Int
end

ProfilePoint(value, loss, params, ret, counter) =
    ProfilePoint(value, loss, params, ret, counter, 0)

"""
    struct EndPoint
        value::Union{Real, Nothing}           # value of endpoint or `nothing`
        profilePoints::Array{ProfilePoint, 1} # vector of profile points
        status::Symbol                        # CICOBase endpoint result code
        direction::Symbol                     # `:right` or `:left`
        counter::Int                          # number of `loss_func` evaluations
        supreme::Union{Real, Nothing}         # maximal value inside profile interval
    end

Structure storing one endpoint of a confidence interval. `status` is a
CICOBase result code, distinct from the NLopt code in `ProfilePoint.ret`:

- `:BORDER_FOUND_BY_SCAN_TOL`: the method returned an endpoint estimate after
  meeting its scan tolerance. This does not guarantee an exact loss boundary.
- `:BORDER_FOUND_BY_LOSS_TOL`: an extrapolation method found a profile loss
  within `loss_tol` of the critical level.
- `:SCAN_BOUND_REACHED`: the scan limit was reached while the loss was below
  the critical level; no endpoint was found within that limit.
- `:MAX_ITER_STOP`: the evaluation limit was reached before an endpoint was
  found. The limit concerns evaluations, despite the status name.
- `:NLOPT_FORCED_STOP`: NLopt was stopped without a recognized reason code
  (including an explicit `NLopt.ForcedStop()`).
- `:LOSS_ERROR_STOP`: `loss_func` threw during evaluation.
- `:SCAN_ERROR_STOP`: `scan_func` threw during evaluation.
- `:LOSS_GRAD_ERROR_STOP`: the supplied or computed loss gradient threw.
- `:SCAN_GRAD_ERROR_STOP`: the supplied or computed scan gradient threw.
- `:NLOPT_FAILURE`: NLopt reported a failure; the numerical cause is unknown.
- `:UNKNOWN_STOP`: an unclassified exception or NLopt result occurred.

Errors outside these optimization paths may still propagate as exceptions.
"""
struct EndPoint
    value::Union{Real, Nothing}
    profilePoints::Array{ProfilePoint, 1}
    status::Symbol
    direction::Symbol
    counter::Int
    supreme::Union{Real, Nothing}
end
