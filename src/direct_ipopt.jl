"""
Experimental direct constrained optimization with Ipopt (no AUGLAG).

`scan_grad` and `loss_grad` follow the common endpoint interface. Set
`hessian=:AUTODIFF` for exact derivatives, a function for a supplied loss
Hessian, or the default `:LIMITED_MEMORY`. At this low-level entry point a
supplied Hessian must use optimization coordinates.
"""
function get_right_endpoint(
    theta_init::Vector{Float64},
    scan_func::Function,
    loss_func::Function,
    ::Val{:DIRECT_IPOPT};
    theta_bounds::Vector{Tuple{Float64,Float64}} = fill(DEFAUL_THETA_BOUNDS_ITEM, length(theta_init)),
    scan_bound::Float64 = 9.0,
    scan_tol::Float64 = 1e-3,
    loss_tol::Float64 = 1e-3,
    max_iter::Int = 10^5,
    scan_grad::Union{Function,Symbol} = :EMPTY,
    loss_grad::Union{Function,Symbol} = :EMPTY,
    hessian::Union{Function,Symbol} = :LIMITED_MEMORY,
    scan_linear::Bool = false,
    kwargs...
)
    scan_tol > 0 && isfinite(scan_tol) || throw(ArgumentError("scan_tol must be positive and finite for Ipopt"))
    loss_tol > 0 && isfinite(loss_tol) || throw(ArgumentError("loss_tol must be positive and finite for Ipopt"))
    (hessian isa Function || hessian in (:LIMITED_MEMORY, :AUTODIFF)) ||
        throw(ArgumentError("hessian must be a function, :LIMITED_MEMORY or :AUTODIFF"))
    haskey(kwargs, :limited_memory) &&
        throw(ArgumentError("use hessian=:LIMITED_MEMORY or :AUTODIFF"))
    for (name, grad) in ((:scan_grad, scan_grad), (:loss_grad, loss_grad))
        (grad isa Function || grad in (:AUTODIFF, :FINITE)) ||
            throw(ArgumentError("$name must be a function, :AUTODIFF or :FINITE for Ipopt"))
    end
    n = length(theta_init)
    failure = Ref{Union{Nothing,Symbol}}(nothing)
    function evaluate(f, x, status)
        try
            return f(x)
        catch err
            failure[] = status
            @warn "Ipopt callback failed" status exception=(err, catch_backtrace())
            return NaN
        end
    end
    function gradient!(out, f, grad, x, status)
        try
            if grad isa Function
                out .= grad(x)
            elseif grad == :AUTODIFF
                ForwardDiff.gradient!(out, f, x)
            else
                Calculus.finite_difference!(f, x, out, :central)
            end
        catch err
            failure[] = status
            @warn "Ipopt gradient callback failed" status exception=(err, catch_backtrace())
            fill!(out, NaN)
        end
        return nothing
    end
    eval_f(x) = -evaluate(scan_func, x, :SCAN_ERROR_STOP)
    function eval_grad_f(x, grad)
        gradient!(grad, scan_func, scan_grad, x, :SCAN_GRAD_ERROR_STOP)
        grad .*= -1
        return nothing
    end
    function eval_g(x, g)
        g[1] = evaluate(loss_func, x, :LOSS_ERROR_STOP)
        g[2] = evaluate(scan_func, x, :SCAN_ERROR_STOP)
        return nothing
    end
    function eval_jac_g(x, rows, cols, values)
        if values === nothing
            for j in 1:n
                rows[j] = 1; cols[j] = j
                rows[n+j] = 2; cols[n+j] = j
            end
        else
            tmp = zeros(n)
            gradient!(tmp, loss_func, loss_grad, x, :LOSS_GRAD_ERROR_STOP)
            values[1:n] .= tmp
            gradient!(tmp, scan_func, scan_grad, x, :SCAN_GRAD_ERROR_STOP)
            values[n+1:2n] .= tmp
        end
        return nothing
    end
    function eval_h(x, rows, cols, obj_factor, lambda, values)
        if values === nothing
            k = 0
            for col in 1:n, row in col:n
                k += 1
                rows[k] = row
                cols[k] = col
            end
        else
            try
                # Objective is -scan_func; constraints are loss_func and scan_func.
                # For a parameter endpoint scan_func is linear in these coordinates.
                h = if hessian isa Function
                    loss_h = hessian(x)
                    size(loss_h) == (n, n) || throw(DimensionMismatch("hessian must be $n by $n"))
                    result = lambda[1] .* loss_h
                    if !scan_linear
                        result .+= (lambda[2] - obj_factor) .* ForwardDiff.hessian(scan_func, x)
                    end
                    result
                else
                    lagrangian(y) = lambda[1] * loss_func(y) +
                        (scan_linear ? zero(eltype(y)) : (lambda[2] - obj_factor) * scan_func(y))
                    ForwardDiff.hessian(lagrangian, x)
                end
                k = 0
                for col in 1:n, row in col:n
                    k += 1
                    values[k] = h[row, col]
                end
            catch err
                failure[] = :UNKNOWN_STOP
                @warn "Ipopt Hessian callback failed" exception=(err, catch_backtrace())
                fill!(values, NaN)
            end
        end
        return nothing
    end
    prob = Ipopt.CreateIpoptProblem(n, first.(theta_bounds), last.(theta_bounds),
        2, [-Inf, -Inf], [0.0, scan_bound], 2n,
        hessian == :LIMITED_MEMORY ? 0 : div(n * (n + 1), 2),
        eval_f, eval_g, eval_grad_f, eval_jac_g, hessian == :LIMITED_MEMORY ? nothing : eval_h)
    prob.x = copy(theta_init)
    Ipopt.AddIpoptStrOption(prob, "hessian_approximation", hessian == :LIMITED_MEMORY ? "limited-memory" : "exact")
    Ipopt.AddIpoptIntOption(prob, "print_level", 0)
    Ipopt.AddIpoptIntOption(prob, "max_iter", max_iter)
    # Ipopt's tol is a scaled optimality criterion, not a tolerance in scan units.
    Ipopt.AddIpoptNumOption(prob, "tol", scan_tol)
    Ipopt.AddIpoptNumOption(prob, "constr_viol_tol", loss_tol)
    Ipopt.AddIpoptStrOption(prob, "sb", "yes")
    ret = try
        Ipopt.IpoptSolve(prob)
    catch err
        @warn "Ipopt solve failed" exception=(err, catch_backtrace())
        return (nothing, ProfilePoint[], something(failure[], :UNKNOWN_STOP))
    end
    failure[] !== nothing && return (nothing, ProfilePoint[], failure[])
    ret == -1 && return (nothing, ProfilePoint[], :MAX_ITER_STOP)
    # 0 = Solve_Succeeded, 1 = Solved_To_Acceptable_Level.
    if !(ret in (0, 1))
        @warn "Unclassified Ipopt termination" ipopt_status=ret
        return (nothing, ProfilePoint[], :UNKNOWN_STOP)
    end
    x = copy(prob.x)
    value = evaluate(scan_func, x, :SCAN_ERROR_STOP)
    loss = evaluate(loss_func, x, :LOSS_ERROR_STOP)
    failure[] !== nothing && return (nothing, ProfilePoint[], failure[])
    if value >= scan_bound - scan_tol && loss < -loss_tol
        return (nothing, ProfilePoint[], :SCAN_BOUND_REACHED)
    end
    point = ProfilePoint(value, loss, x, ret == 0 ? :IPOPT_SOLVED : :IPOPT_ACCEPTABLE, nothing)
    return (value, [point], :BORDER_FOUND_BY_SCAN_TOL)
end

function get_right_endpoint(
    theta_init::Vector{Float64},
    theta_num::Int,
    loss_func::Function,
    method::Val{:DIRECT_IPOPT};
    kwargs...
)
    1 <= theta_num <= length(theta_init) || throw(DomainError(theta_num, "theta_num exceeds theta dimension"))
    scan_func(theta::Vector) = theta[theta_num]
    scan_grad(theta::Vector) = [i == theta_num ? 1.0 : 0.0 for i in eachindex(theta)]
    return get_right_endpoint(theta_init, scan_func, loss_func, method; kwargs..., scan_grad, scan_linear=true)
end
