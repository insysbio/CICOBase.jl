function get_right_endpoint(
    theta_init::Vector{Float64},
    scan_func::Function,
    loss_func::Function,
    method::Val{:DIRECT_SLSQP};
    theta_bounds::Vector{Tuple{Float64,Float64}} = fill(DEFAUL_THETA_BOUNDS_ITEM, length(theta_init)),
    scan_bound::Float64 = 9.0,
    scan_tol::Float64 = 1e-3,
    scan_rtol::Float64 = 0.0,
    loss_tol::Float64 = 1e-3,
    max_iter::Int = 10^5,
    scan_grad::Union{Function,Symbol} = :EMPTY,
    loss_grad::Union{Function,Symbol} = :EMPTY,
    kwargs...
)
    scan_grad == :EMPTY && throw(ArgumentError("`scan_grad` must be set for DIRECT_SLSQP"))
    loss_grad == :EMPTY && throw(ArgumentError("`loss_grad` must be set for DIRECT_SLSQP"))
    isfinite(loss_tol) && loss_tol >= 0 ||
        throw(ArgumentError("loss_tol must be finite and nonnegative"))
    haskey(kwargs, :constraint_tol) &&
        throw(ArgumentError("constraint_tol was replaced by loss_tol"))

    opt = Opt(:LD_SLSQP, length(theta_init))
    opt.lower_bounds = first.(theta_bounds)
    opt.upper_bounds = last.(theta_bounds)
    ftol_abs!(opt, scan_tol)
    ftol_rel!(opt, scan_rtol)
    maxeval!(opt, max_iter)

    function stop_after_callback_error(error, code, grad, source)
        error isa NLopt.ForcedStop && throw(error)
        @warn "Error in $source" exception = (error, catch_backtrace())
        NLopt.force_stop!(opt, code)
        fill!(grad, 0.0)
        return 0.0
    end

    function objective(x, grad)
        if !isempty(grad)
            try
                if scan_grad isa Function
                    grad .= scan_grad(x)
                elseif scan_grad == :AUTODIFF
                    ForwardDiff.gradient!(grad, scan_func, x)
                elseif scan_grad == :FINITE
                    Calculus.finite_difference!(scan_func, x, grad, :central)
                else
                    throw(ArgumentError("unsupported scan_grad: $scan_grad"))
                end
            catch error
                return stop_after_callback_error(error, STOP_CODE_SCAN_GRAD_ERROR, grad, "scan gradient")
            end
        end
        return try
            scan_func(x)
        catch error
            stop_after_callback_error(error, STOP_CODE_SCAN_ERROR, grad, "scan_func")
        end
    end

    function constraint(x, grad)
        value = try
            loss_func(x)
        catch error
            return stop_after_callback_error(error, STOP_CODE_LOSS_ERROR, grad, "loss_func")
        end
        within_bounds = all(i -> theta_bounds[i][1] <= x[i] <= theta_bounds[i][2], eachindex(x))
        if value < 0 && within_bounds
            scan_value = try
                scan_func(x)
            catch error
                return stop_after_callback_error(error, STOP_CODE_SCAN_ERROR, grad, "scan_func")
            end
            if scan_value >= scan_bound
                NLopt.force_stop!(opt, STOP_CODE_SCAN_BOUND)
                fill!(grad, 0.0)
                return value
            end
        end
        if !isempty(grad)
            try
                if loss_grad isa Function
                    grad .= loss_grad(x)
                elseif loss_grad == :AUTODIFF
                    ForwardDiff.gradient!(grad, loss_func, x)
                elseif loss_grad == :FINITE
                    Calculus.finite_difference!(loss_func, x, grad, :central)
                else
                    throw(ArgumentError("unsupported loss_grad: $loss_grad"))
                end
            catch error
                return stop_after_callback_error(error, STOP_CODE_LOSS_GRAD_ERROR, grad, "loss gradient")
            end
        end
        return value
    end

    max_objective!(opt, objective)
    inequality_constraint!(opt, constraint, loss_tol)
    optf, optx, ret = try
        optimize(opt, theta_init)
    catch error
        @warn "Error in DIRECT_SLSQP optimization" exception = (error, catch_backtrace())
        return (nothing, ProfilePoint[], :UNKNOWN_STOP)
    end

    code = NLopt.force_stop(opt)
    code == STOP_CODE_SCAN_BOUND && return (nothing, ProfilePoint[], :SCAN_BOUND_REACHED)
    code == STOP_CODE_LOSS_ERROR && return (nothing, ProfilePoint[], :LOSS_ERROR_STOP)
    code == STOP_CODE_SCAN_ERROR && return (nothing, ProfilePoint[], :SCAN_ERROR_STOP)
    code == STOP_CODE_LOSS_GRAD_ERROR && return (nothing, ProfilePoint[], :LOSS_GRAD_ERROR_STOP)
    code == STOP_CODE_SCAN_GRAD_ERROR && return (nothing, ProfilePoint[], :SCAN_GRAD_ERROR_STOP)
    ret == :MAXEVAL_REACHED && return (nothing, ProfilePoint[], :MAX_ITER_STOP)
    ret == :FORCED_STOP && return (nothing, ProfilePoint[], :NLOPT_FORCED_STOP)
    ret == :FAILURE && return (nothing, ProfilePoint[], :NLOPT_FAILURE)
    if ret == :FTOL_REACHED
        loss = try
            loss_func(optx)
        catch error
            @warn "Error in loss_func after DIRECT_SLSQP" exception = (error, catch_backtrace())
            return (nothing, ProfilePoint[], :LOSS_ERROR_STOP)
        end
        point = ProfilePoint(optf, loss, optx, ret, nothing)
        return (optf, [point], :BORDER_FOUND_BY_SCAN_TOL)
    end
    @warn "Unclassified DIRECT_SLSQP termination" ret
    return (nothing, ProfilePoint[], :UNKNOWN_STOP)
end

function get_right_endpoint(
    theta_init::Vector{Float64},
    theta_num::Int,
    loss_func::Function,
    method::Val{:DIRECT_SLSQP};
    kwargs...
)
    1 <= theta_num <= length(theta_init) || throw(DomainError(theta_num, "theta_num exceeds theta dimension"))
    scan_func(theta::Vector) = theta[theta_num]
    scan_grad(theta::Vector) = [i == theta_num ? 1.0 : 0.0 for i in eachindex(theta)]
    return get_right_endpoint(theta_init, scan_func, loss_func, method; kwargs..., scan_grad)
end
