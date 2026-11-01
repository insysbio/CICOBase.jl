# evaluate right bound of scan_func
function get_right_endpoint(
    theta_init::Vector{Float64}, # initial point of parameters
    scan_func::Function, # h(theta) function for predictions or parameters
    loss_func::Function, # lambda(theta) - labmbda_min - delta_lambda
    method::Val{:CICO_ONE_PASS}; # function works only for method ONE_PASS;

    theta_bounds::Vector{Tuple{Float64,Float64}} = fill(DEFAUL_THETA_BOUNDS_ITEM, length(theta_init)),
    scan_bound::Float64 = 9.0,
    scan_tol::Float64 = 1e-3,
    scan_rtol::Float64 = 0.,
    loss_tol::Float64 = DEFAULT_LOSS_TOL, # we have no idea how to implement loss_tol
    # better results for LN_NELDERMEAD, :LD_MMA, :LD_SLSQP, :LD_CCSAQ
    # worse results for :LN_COBYLA, :LN_PRAXIS
    # errors for :LN_BOBYQA, :LN_SBPLX, :LN_NEWUOA
    local_alg::Symbol = DEFAULT_LOCAL_ALG,
    local_initial_step = nothing,
    # options for local fitter :max_iter
    max_iter::Int = 10^5,
    scan_grad::Union{Function, Symbol} = :EMPTY,
    loss_grad::Union{Function, Symbol} = :EMPTY,
    #kwargs...
)
    # dim of the theta vector
    n_theta = length(theta_init)
    
    # checking scan_grad, loss_grad
    is_gradient = occursin(r"^LD_", String(local_alg))
    if scan_grad == :EMPTY && is_gradient
        throw(ArgumentError("`scan_grad` must be set for gradient local fitter `$(local_alg)`"))
    end
    if loss_grad == :EMPTY && is_gradient
        throw(ArgumentError("`loss_grad` must be set for gradient local fitter `$(local_alg)`"))
    end

    # optimizer
    local_opt = Opt(local_alg, n_theta)
    # The local objective includes AUGLAG's penalty for loss_func. Its scale
    # changes with the penalty coefficient, so local ftol_abs is not in pure
    # scan_func units and local ftol_rel measures a different relative change.
    # Keep the old values for now; tune both local tolerances independently.
    local_ftol_abs = scan_tol
    local_ftol_rel = scan_rtol
    ftol_abs!(local_opt, local_ftol_abs)
    ftol_rel!(local_opt, local_ftol_rel)
    # AUGLAG replaces a nonpositive local xtol_rel with 1e-8 when local
    # ftol_rel is also nonpositive. Keep this criterion nearly inactive so
    # the absolute function tolerance can stop the local optimization first.
    xtol_rel!(local_opt, 1e-18)
    # initial step for optimization
    if !isnothing(local_initial_step)
        initial_step!(local_opt, _validated_local_initial_step(local_initial_step, n_theta))
    end

    NLopt.nlopt_set_param(local_opt, "tolg", MAGIC_TOLG) # default 1e-8

    opt = Opt(:LN_AUGLAG, n_theta)
    function stop_after_callback_error(e, code, g, source)
        e isa NLopt.ForcedStop && throw(e) # Preserve an explicit NLopt stop from user code.
        @warn "Error in $source" exception = (e, catch_backtrace())
        NLopt.force_stop!(opt, code)
        fill!(g, 0.0)
        return 0.0 # NLopt still needs a finite callback result before stopping.
    end

    function constraints_func(x, g) # testing grad methods
        loss_value = try
            loss_func(x)
        catch e
            return stop_after_callback_error(e, STOP_CODE_LOSS_ERROR, g, "loss_func")
        end
        
        if loss_value < 0.
            scan_value = try
                scan_func(x)
            catch e
                return stop_after_callback_error(e, STOP_CODE_SCAN_ERROR, g, "scan_func")
            end
            if scan_value >= scan_bound
                NLopt.force_stop!(opt, STOP_CODE_SCAN_BOUND)
                fill!(g, 0.0)
                return loss_value
            end
        end
        if length(g) > 0
            try
                if isa(loss_grad, Function)
                    g .= loss_grad(x)
                elseif loss_grad == :AUTODIFF
                    ForwardDiff.gradient!(g, loss_func, x)
                elseif loss_grad == :FINITE
                    Calculus.finite_difference!(loss_func, x, g, :central)
                end
            catch e
                return stop_after_callback_error(e, STOP_CODE_LOSS_GRAD_ERROR, g, "loss gradient")
            end
        end
        
        return loss_value
    end

    function obj_func(x, g)
        if length(g) > 0
            try
                if isa(scan_grad, Function)
                    g .= scan_grad(x)
                elseif scan_grad == :AUTODIFF
                    ForwardDiff.gradient!(g, scan_func, x)
                elseif scan_grad == :FINITE
                    Calculus.finite_difference!(scan_func, x, g, :central)
                end
            catch e
                return stop_after_callback_error(e, STOP_CODE_SCAN_GRAD_ERROR, g, "scan gradient")
            end
        end
        return try
            scan_func(x)
        catch e
            stop_after_callback_error(e, STOP_CODE_SCAN_ERROR, g, "scan_func")
        end
    end

    # constrain optimizer
    ftol_abs!(opt, scan_tol)
    ftol_rel!(opt, scan_rtol)

    max_objective!(
        opt,
        obj_func
        )
        
    local_optimizer!(opt, local_opt)
    maxeval!(opt, max_iter)

    # inequality constraints
    inequality_constraint!(
        opt,
        constraints_func,
        MAGIC_CONSTRAINT_TOL
    )

    # version 1: box constrains from :LN_AUGLAG send to specific methods
    opt.lower_bounds = [tb[1] for tb in theta_bounds]
    opt.upper_bounds = [tb[2] for tb in theta_bounds]

    # version 2: creating lower and upper bounds as general constraints
    #= 
    [ inequality_constraint!(
        opt,
        (x, g) -> x[i] - theta_bounds[i][2],
        0.
    ) for i in 1:n_theta ]
    [ inequality_constraint!(
        opt,
        (x, g) -> theta_bounds[i][1] - x[i],
        0.
    ) for i in 1:n_theta ]
    =#

    # start optimization
    (optf, optx, ret) = try 
        optimize(opt, theta_init)
    catch e
        @warn "Error when call optimize($opt, $theta_init)" exception = (e, catch_backtrace())
        return (nothing, ProfilePoint[], :UNKNOWN_STOP)
    end

    stop_code = NLopt.force_stop(opt)
    if stop_code == STOP_CODE_SCAN_BOUND
        res = (nothing, ProfilePoint[], :SCAN_BOUND_REACHED)
    elseif stop_code == STOP_CODE_LOSS_ERROR
        res = (nothing, ProfilePoint[], :LOSS_ERROR_STOP)
    elseif stop_code == STOP_CODE_SCAN_ERROR
        res = (nothing, ProfilePoint[], :SCAN_ERROR_STOP)
    elseif stop_code == STOP_CODE_LOSS_GRAD_ERROR
        res = (nothing, ProfilePoint[], :LOSS_GRAD_ERROR_STOP)
    elseif stop_code == STOP_CODE_SCAN_GRAD_ERROR
        res = (nothing, ProfilePoint[], :SCAN_GRAD_ERROR_STOP)
    elseif ret == :MAXEVAL_REACHED
        res = (nothing, ProfilePoint[], :MAX_ITER_STOP)
    elseif ret == :FORCED_STOP
        res = (nothing, ProfilePoint[], :NLOPT_FORCED_STOP)
    elseif ret == :FAILURE
        res = (nothing, ProfilePoint[], :NLOPT_FAILURE)
    elseif ret == :FTOL_REACHED # successful result
        loss = try
            loss_func(optx)
        catch e
            @warn "Error in loss_func after optimization" exception = (e, catch_backtrace())
            return (nothing, ProfilePoint[], :LOSS_ERROR_STOP)
        end
        pp = [ ProfilePoint(optf, loss, optx, ret, nothing) ]
        res = (optf, pp, :BORDER_FOUND_BY_SCAN_TOL)
    else
        # Other NLopt results without a recognized stopping condition.
        res = (nothing, ProfilePoint[], :UNKNOWN_STOP)
    end

    return res
end

function get_right_endpoint(
    theta_init::Vector{Float64}, # initial point of parameters
    theta_num::Int, # number of parameter to scan
    loss_func::Function, # lambda(theta) - labmbda_min - delta_lambda
    method::Val{:CICO_ONE_PASS}; # function works only for method ONE_PASS;

    theta_bounds::Vector{Tuple{Float64,Float64}} = fill(DEFAUL_THETA_BOUNDS_ITEM, length(theta_init)),
    scan_bound::Float64 = 9.0,
    scan_tol::Float64 = 1e-3,
    scan_rtol::Float64 = 0.,
    loss_tol::Float64 = DEFAULT_LOSS_TOL,
    local_alg::Symbol = DEFAULT_LOCAL_ALG,
    local_initial_step = nothing,
    max_iter::Int = 10^5,
    loss_grad::Union{Function, Symbol} = :EMPTY,
    #kwargs...
)
    # checking arguments
    if theta_num > length(theta_init)
        throw(DomainError(theta_num, "theta_num exceed theta dimension"))
    end

    scan_func(theta::Vector) = theta[theta_num]
    scan_grad(theta::Vector) = begin
        res = zeros(length(theta_init))
        res[theta_num] = 1.
        return res
    end

    get_right_endpoint(
        theta_init,
        scan_func,
        loss_func,
        method;

        theta_bounds,
        scan_bound,
        scan_tol,
        scan_rtol,
        loss_tol,
        local_alg,
        local_initial_step,
        max_iter,
        scan_grad,
        loss_grad
        #kwargs...
    )
end
