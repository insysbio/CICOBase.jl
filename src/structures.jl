"""
    struct ProfilePoint
        value::Float64               # x value of profile point
        loss::Float64                # y value of profile point (loss function value)
        params::Array{Float64, 1}    # vector of optimal values of `loss_func` arguments
        ret::Symbol                  # `NLopt.optimize()` result code
        counter::Union{Int, Nothing} # number of `loss_func` evaluations
    end

Structure storing one point from a profile function. `ret` is the result code
returned by NLopt, or `:OPTIMIZATION_SKIPPED` when no optimization was run.
Common NLopt codes are:

- `:FTOL_REACHED`: the optimizer met its objective-value tolerance; this alone
  does not establish that the loss reached its critical value.
- `:MAXEVAL_REACHED`: the optimizer reached its evaluation limit.
- `:FORCED_STOP`: the optimizer was stopped externally. During profile
  optimization this often follows an exception in `loss_func`, which `_profile`
  converts to `NLopt.ForcedStop()`, but other forced stops are possible.

Other NLopt result codes are preserved in `ret`.

"""
struct ProfilePoint
    value::Float64
    loss::Float64
    params::Array{Float64, 1}
    ret::Symbol
    counter::Union{Int, Nothing}
end

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
  meeting its scan tolerance. In `:CICO_ONE_PASS` this follows NLopt's
  `:FTOL_REACHED`; extrapolation methods use the change in the estimated
  endpoint. It does not guarantee an exact loss boundary.
- `:BORDER_FOUND_BY_LOSS_TOL`: an extrapolation method found a profile loss
  within `loss_tol` of the critical level.
- `:SCAN_BOUND_REACHED`: the scan limit was reached while the loss was below
  the critical level; no endpoint was found within that limit.
- `:MAX_ITER_STOP`: NLopt returned `:MAXEVAL_REACHED` before an endpoint was
  found. The limit concerns evaluations, despite the status name.
- `:FORCED_STOP`: NLopt returned `:FORCED_STOP` without a confirmed scan-bound
  stop. In extrapolation methods this often follows a `loss_func` exception,
  but it can have another cause, including an explicit forced stop.
- `:OPTIMIZATION_EXCEPTION_STOP`: in `:CICO_ONE_PASS`, `optimize` threw an
  exception. It can originate in `loss_func`, `scan_func`, a gradient, or NLopt;
  the status alone does not identify the source.
- `:UNKNOWN_STOP`: in `:CICO_ONE_PASS`, NLopt returned a code not handled above,
  such as `:FAILURE` without a confirmed scan-bound stop. The cause is unknown.

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
