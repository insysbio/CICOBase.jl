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

    function constraints_func(x, g) # testing grad methods
        loss_value = loss_func(x)
        
        if (loss_value < 0.) && (scan_func(x) >= scan_bound)
            throw(NLopt.ForcedStop()) # stop optimization because of scan_bound reached
        elseif length(g) > 0
          if isa(loss_grad, Function)
            g .= loss_grad(x)
          elseif loss_grad == :AUTODIFF
            ForwardDiff.gradient!(g, loss_func, x)
          elseif loss_grad == :FINITE
            Calculus.finite_difference!(loss_func,x,g,:central)
          end
        end
        
        return loss_value
    end

    function obj_func(x, g)
        if length(g) > 0
          if isa(scan_grad, Function)
            g .= scan_grad(x)
          elseif scan_grad == :AUTODIFF
            ForwardDiff.gradient!(g, scan_func, x)
          elseif scan_grad == :FINITE
            Calculus.finite_difference!(scan_func, x, g, :central)
          end
        end
        return scan_func(x)
    end

    # constrain optimizer
    opt = Opt(:LN_AUGLAG, n_theta)
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
    #=
    opt.lower_bounds = [tb[1] for tb in theta_bounds]
    opt.upper_bounds = [tb[2] for tb in theta_bounds]
    =#

    # version 2: creating lower and upper bounds as general constraints
     
    [ inequality_constraint!(
        opt,
        (x, g) -> begin
            if !isempty(g)
                fill!(g, 0.0)
                g[i] = 1.0
            end
            x[i] - theta_bounds[i][2]
        end,
        0.
    ) for i in 1:n_theta ]
    [ inequality_constraint!(
        opt,
        (x, g) -> begin
            if !isempty(g)
                fill!(g, 0.0)
                g[i] = -1.0
            end
            theta_bounds[i][1] - x[i]
        end,
        0.
    ) for i in 1:n_theta ]
    

    # start optimization
    # TODO: just throw error if optimization fails, not :LOSS_ERROR_STOP
    (optf, optx, ret) = try 
        optimize(opt, theta_init)
    catch e
        @warn "Error when call optimize($opt, $theta_init)"
        return (nothing, ProfilePoint[], :LOSS_ERROR_STOP)
    end

    if ret == :MAXEVAL_REACHED
        pp = ProfilePoint[]
        res = (nothing, pp, :MAX_ITER_STOP)
    elseif ret == :FORCED_STOP || ret == :FAILURE # successful result
        pp = ProfilePoint[]
        res = (nothing, pp, :SCAN_BOUND_REACHED)
    elseif ret == :FTOL_REACHED # successful result
        loss = loss_func(optx)
        pp = [ ProfilePoint(optf, loss, optx, ret, nothing) ]
        res = (optf, pp, :BORDER_FOUND_BY_SCAN_TOL)
    else
        # this part is reached in case of :FAILURE return code
        # we have seen it in case of gradient methods
        pp = ProfilePoint[]
        res = (nothing, pp, :UNKNOWN_STOP)
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
