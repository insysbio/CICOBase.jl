"""
    validate_interval(interval::ParamInterval; boundary_loss_atol, profile_loss_atol,
                      parameter_atol=1e-8, loss_consistency_atol=1e-8,
                      profile_ftol_abs=1e-10, profile_maxeval=100_000,
                      profile_local_alg=:LN_NELDERMEAD)

Independently check both endpoints of an interval returned by `get_interval`
for a parameter. The two required tolerances are absolute, in loss units:
`boundary_loss_atol` checks the critical level and `profile_loss_atol` limits
the improvement found by minimizing loss with the scanned parameter fixed.
They do not change the original method's `loss_tol` or its result.

`parameter_atol` is an absolute tolerance in original parameter units for
matching the saved profile point and fixed parameter to the endpoint.
`loss_consistency_atol` checks whether a fresh
loss evaluation agrees with the stored value. The remaining options control
the separate, derivative-free profile optimization. It starts at the returned
point and at one nearby point per free coordinate. Defaults are operational
choices, not validated tolerances for a particular model.

Returns a named tuple with `left` and `right` reports and the settings. Each
report has the original status, an `outcome` (`:pass`, `:fail`,
`:inconclusive`, or `:not_checked`), a `reason`, individual `checks`, and
diagnostic values.
The input interval is not modified. A successful profile check means only
that the chosen local restart found no sufficiently better point; it does not
prove global optimality or the globally extreme endpoint.

Only `:BORDER_FOUND_BY_SCAN_TOL` endpoints are validated. For
`:SCAN_BOUND_REACHED` and `:BORDER_FOUND_BY_LOSS_TOL`, the outcome is
`:not_checked`; no loss or profile evaluation is performed. Other termination
statuses are reported as inconclusive without evaluating the loss. Prediction
intervals defined by a general scan function are not supported.
"""
function validate_interval(interval::ParamInterval; # Input data and statuses affect all reasons.
    boundary_loss_atol::Real, # :boundary_loss_outside_tolerance
    profile_loss_atol::Real, # :profile_optimization_inconclusive
    parameter_atol::Real = 1e-8, # :missing_endpoint_point, :profile_optimization_inconclusive
    loss_consistency_atol::Real = 1e-8, # :loss_inconsistent, :profile_optimization_inconclusive
    # repeated identification
    profile_ftol_abs::Real = 1e-10, # :profile_improved, :profile_optimization_inconclusive
    profile_maxeval::Integer = 100_000, # :profile_improved, :profile_optimization_inconclusive
    profile_local_alg::Symbol = :LN_NELDERMEAD, # :profile_improved, :profile_optimization_inconclusive
)
    input = interval.input
    input isa ParamIntervalInput || throw(ArgumentError("validate_interval supports parameter intervals only"))
    for (name, value) in ((:boundary_loss_atol, boundary_loss_atol),
                          (:profile_loss_atol, profile_loss_atol),
                          (:parameter_atol, parameter_atol),
                          (:loss_consistency_atol, loss_consistency_atol),
                          (:profile_ftol_abs, profile_ftol_abs))
        isfinite(value) && value >= 0 || throw(ArgumentError("$name must be finite and nonnegative"))
    end
    profile_ftol_abs > 0 || throw(ArgumentError("profile_ftol_abs must be positive"))
    profile_maxeval > 0 || throw(ArgumentError("profile_maxeval must be positive"))
    startswith(String(profile_local_alg), "LN_") || throw(ArgumentError("profile_local_alg must be a derivative-free local NLopt algorithm"))

    n = length(input.theta_init)
    i = input.theta_num
    1 <= i <= n || throw(ArgumentError("invalid parameter index"))
    options = input.options
    scale = get(options, :scale, fill(:direct, n))
    length(scale) == n || throw(DimensionMismatch("scale length must match parameters"))
    bounds = get(options, :theta_bounds, unscaling.(fill(DEFAUL_THETA_BOUNDS_ITEM, n), scale))
    length(bounds) == n || throw(DimensionMismatch("theta_bounds length must match parameters"))
    loss_crit = get(options, :loss_crit, 0.0)
    isfinite(loss_crit) || throw(ArgumentError("loss_crit must be finite"))
    settings = (; boundary_loss_atol, profile_loss_atol, parameter_atol,
        loss_consistency_atol, profile_ftol_abs, profile_maxeval, profile_local_alg)

    valid_params(p) = length(p) == n && all(isfinite, p) &&
        all(j -> (scale[j] != :log || p[j] > 0) &&
            (scale[j] != :logit || 0 < p[j] < 1), 1:n)
    function eval_loss(p)
        try
            v = input.loss_func(copy(p))
            return v isa Real && isfinite(v) ? (value=Float64(v), error=nothing) :
                (value=nothing, error="loss is non-real or non-finite")
        catch e
            e isa InterruptException && rethrow()
            return (value=nothing, error=sprint(showerror, e))
        end
    end
    function report(ep; outcome=:inconclusive, reason=:not_checked, checks=(;),
                    point=nothing, loss=nothing, loss_error=nothing,
                    boundary_residual=nothing, profile=nothing)
        (; direction=ep.direction, original_status=ep.status, endpoint=ep.value,
           outcome, reason, checks, point, loss, loss_error, boundary_residual, profile)
    end
    function validate_endpoint(ep, direction)
        ep.status == :SCAN_BOUND_REACHED && return report(ep; outcome=:not_checked, reason=:scan_bound_not_validated)
        ep.status == :BORDER_FOUND_BY_LOSS_TOL && return report(ep; outcome=:not_checked, reason=:status_not_validated)
        ep.direction == direction || return report(ep; reason=:invalid_direction)
        ep.status == :BORDER_FOUND_BY_SCAN_TOL ||
            return report(ep; reason=:unsupported_or_unsuccessful_status)
        if isnothing(ep.value) || !isfinite(ep.value)
            return report(ep; reason=:missing_endpoint)
        end

        # All matching points are retained: a later profile refinement may have
        # appended points after the original endpoint.
        matches = filter(ep.profilePoints) do p
            length(p.params) == n && isfinite(p.value) && isfinite(p.params[i]) &&
            abs(p.value - p.params[i]) <= parameter_atol &&
            abs(p.value - ep.value) <= parameter_atol && abs(p.params[i] - ep.value) <= parameter_atol
        end
        isempty(matches) && return report(ep; reason=:missing_endpoint_point)
        # Prefer the nearest point; ties retain the first original point.
        matches = [matches[argmin([abs(p.value - ep.value) for p in matches])]]
        last_report = nothing
        uncertain_report = nothing
        for point in matches
            valid_params(point.params) || begin
                last_report = report(ep; reason=:invalid_parameters, point=point)
                continue
            end
            evaluated = eval_loss(point.params)
            if isnothing(evaluated.value)
                last_report = report(ep; reason=:loss_evaluation_failed, point=point, loss_error=evaluated.error)
                uncertain_report = last_report
                continue
            end
            loss = evaluated.value
            consistent = isfinite(point.loss) && abs(loss - point.loss) <= loss_consistency_atol
            if !consistent
                last_report = report(ep; reason=:loss_inconsistent, point=point, loss=loss,
                                     checks=(loss_consistent=false,))
                uncertain_report = last_report
                continue
            end
            residual = abs(loss - loss_crit)
            boundary_ok = residual <= boundary_loss_atol
            if n == 1
                profile_result = (status=:OPTIMIZATION_SKIPPED, loss=loss,
                    params=copy(point.params), improvement=0.0, evaluations=1, error=nothing)
                return report(ep; outcome=boundary_ok ? :pass : :fail,
                    reason=boundary_ok ? :checks_passed : :boundary_loss_outside_tolerance,
                    point=point, loss=loss, boundary_residual=residual, profile=profile_result,
                    checks=(loss_consistent=true, boundary=boundary_ok, profile=true))
            end
            # profile() optimizes in transformed coordinates and returns its
            # parameters in those coordinates; convert before checking loss.
            profile_result = try
                prof = profile(copy(point.params), i, input.loss_func;
                    scale=scale, theta_bounds=bounds, local_alg=profile_local_alg,
                    ftol_abs=Float64(profile_ftol_abs), maxeval=Int(profile_maxeval))
                scale_mod = copy(scale)
                scale_mod[i] = :direct
                initial = scaling.(point.params, scale_mod)
                internal_bounds = scaling.(bounds, scale_mod)
                starts = [initial]
                for j in 1:n
                    j == i && continue
                    step = 0.1 * max(1.0, abs(initial[j]))
                    lo, hi = internal_bounds[j]
                    candidate = clamp(initial[j] + step, lo, hi)
                    candidate == initial[j] && (candidate = clamp(initial[j] - step, lo, hi))
                    if isfinite(candidate) && candidate != initial[j]
                        start = copy(initial)
                        start[j] = candidate
                        push!(starts, start)
                    end
                end
                runs = map(starts) do start
                    try
                        pp = prof(Float64(ep.value); theta_init_i=copy(start))
                        params = unscaling.(pp.params, scale_mod)
                        fresh = valid_params(params) && abs(params[i] - ep.value) <= parameter_atol ?
                            eval_loss(params) : (value=nothing, error="invalid optimized parameters")
                        consistent = !isnothing(fresh.value) && isfinite(pp.loss) &&
                            abs(fresh.value - pp.loss) <= loss_consistency_atol
                        (status=pp.ret, loss=consistent ? fresh.value : nothing, params=params,
                         evaluations=pp.counter, error=consistent ? nothing :
                             something(fresh.error, "optimized loss is inconsistent"))
                    catch e
                        e isa InterruptException && rethrow()
                        (status=:EXCEPTION, loss=nothing, params=nothing,
                         evaluations=nothing, error=sprint(showerror, e))
                    end
                end
                valid_runs = filter(run -> !isnothing(run.loss), runs)
                best = isempty(valid_runs) ? nothing : valid_runs[argmin([run.loss for run in valid_runs])]
                accepted = all(run -> run.status in (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED, :STOPVAL_REACHED) &&
                               !isnothing(run.loss), runs)
                (status=accepted ? :CONVERGED : :INCONCLUSIVE,
                 loss=isnothing(best) ? nothing : best.loss,
                 params=isnothing(best) ? nothing : best.params,
                 improvement=isnothing(best) ? nothing : max(0.0, loss - best.loss),
                 evaluations=sum(run -> something(run.evaluations, 0), runs), error=nothing, runs=runs)
            catch e
                e isa InterruptException && rethrow()
                (status=:EXCEPTION, loss=nothing, params=nothing, improvement=nothing,
                 evaluations=nothing, error=sprint(showerror, e))
            end
            improved = !isnothing(profile_result.improvement) && profile_result.improvement > profile_loss_atol
            accepted = profile_result.status == :CONVERGED &&
                !isnothing(profile_result.loss)
            profile_ok = improved ? false : (accepted ? true : nothing)
            outcome = !boundary_ok || improved ? :fail : (accepted ? :pass : :inconclusive)
            reason = !boundary_ok ? :boundary_loss_outside_tolerance :
                improved ? :profile_improved : accepted ? :checks_passed : :profile_optimization_inconclusive
            return report(ep; outcome, reason, point=point, loss=loss, boundary_residual=residual,
                profile=profile_result, checks=(loss_consistent=true,
                boundary=boundary_ok, profile=profile_ok))
        end
        return isnothing(uncertain_report) ? last_report : uncertain_report
    end
    left = validate_endpoint(interval.result[1], :left)
    right = validate_endpoint(interval.result[2], :right)
    return (; left, right, settings)
end
