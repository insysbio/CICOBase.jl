"""
    logit10(x::Real)
Function transforming interval [0,1] to [-Inf, Inf] using logit transformation.
"""
function logit10(x::Real)
    log10(x / (1.0 - x))
end

"""
    logistic10(x::Real)
Function transforming interval [-Inf, Inf] to [0,1] using logistic transformation.
Inverse function for [`logit10`](@ref).
"""
function logistic10(x::Real)
    exp10(x) / (exp10(x) + 1.0)
end

"""
    scaling(x::Real, scale::Symbol = :direct)
Transforms values from specific scale to range [-Inf, Inf] based on option.

## Return
Transformed value.

## Arguments
* `x`: input value.
* `scale`: transformation type: `:direct` (`:lin`), `:log`, `:logit`.
"""
function scaling(x::Real, scale::Symbol = :direct)
    if scale == :direct || scale == :lin
        return x
    elseif scale == :log
        return log10(x)
    elseif scale == :logit
        return logit10(x)
    else
        throw(DomainError(scale, "scale type is not supported"))
    end
end

"""
    unscaling(x::Real, scale::Symbol = :direct)
Transforms values from [-Inf, Inf] to specific scale based on option. Inverse function
for [`scaling`](@ref).

## Return
Transformed value.

## Arguments
* `x`: input value.
* `scale`: transformation type: `:direct` (`:lin`), `:log`, `:logit`.
"""
function unscaling(x::Real, scale::Symbol = :direct)
    if scale == :direct || scale == :lin
        return x
    elseif scale == :log
        return exp10(x)
    elseif scale == :logit
        return logistic10(x)
    else
        throw(DomainError(scale, "scale type is not supported"))
    end
end

scaling(x_tup::Tuple{Float64,Float64}, scale::Symbol = :direct) = scaling.(x_tup, scale)

unscaling(x_tup::Tuple{Float64,Float64}, scale::Symbol = :direct) = unscaling.(x_tup, scale)

scaling(::Nothing, ::Symbol) = nothing

unscaling(::Nothing, ::Symbol) = nothing

"""
    function get_endpoint(
        theta_init::Vector{Float64},
        theta_num::Int,
        loss_func::Function,
        method::Symbol,
        direction::Symbol = :right;

        loss_crit::Float64 = 0.0,
        scale::Vector{Symbol} = fill(:direct, length(theta_init)),
        theta_bounds::Vector{Tuple{Float64,Float64}} = unscaling.(
            fill((-Inf, Inf), length(theta_init)),
            scale
            ),
        scan_bound::Float64 = unscaling(
            (direction==:left) ? -9.0 : 9.0,
            scale[theta_num]
            ),
        scan_tol::Float64 = 1e-3,
        loss_tol::Float64 = 0.,
        local_alg::Symbol = :LN_NELDERMEAD,
        max_iter::Int = 10^5,
        loss_grad::Union{Function, Symbol} = :EMPTY,
        silent::Bool = false
    )

Calculates confidence interval's right or left endpoints for a given parameter `theta_num`.

## Return
[`EndPoint`](@ref) object storing confidence interval's endpoint and intermediate profile points.

## Arguments
- `theta_init`: starting values of parameter vector ``\\theta``. The starting values should not necessary be the optimum values of `loss_func` but `loss_func(theta_init)` should be lower than `loss_crit`.
- `theta_num`: index of vector component for identification: `theta_init(theta_num)`.
- `loss_func`: loss function ``\\Lambda\\left(\\theta\\right)`` for profile likelihood-based (PL) identification. Usually we use log-likelihood for PL analysis: ``\\Lambda( \\theta ) = - 2 ln\\left( L(\\theta) \\right)``.
- `method`: computational method to estimate confidence interval's endpoint. Currently the following methods are implemented: `:CICO_ONE_PASS`, `:LIN_EXTRAPOL`, `:QUADR_EXTRAPOL`.
- `direction`: `:right` or `:left` endpoint to estimate.

## Keyword arguments
- `loss_crit` : critical level of loss function. Confidence interval's endpoint value is the intersection point of profile likelihood and `loss_crit` level.
- `scale` : vector of scale transformations for each parameters' component. Possible values: `:direct` (`:lin`), `:log`, `:logit`. This option can speed up the optimization, especially for wide `theta_bounds`. The default value is `:direct` (no transformation) for all parameters.
- `theta_bounds` : vector of tuple `(lower_bound, upper_bound)` for each parameter. Bounds define the ranges for possible parameter values. Default bounds are `(-Inf,Inf)`.
- `scan_bound` : value which states the area of confidence point analysis.
- `scan_tol` : Absolute tolerance for `theta_num` parameter used as termination criterion.
- `loss_tol` : Absolute tolerance controlling `loss_func` closeness to `loss_crit` (termination criterion). Currently doesn't work for `:CICO_ONE_PASS` method because of limitation in `LN_AUGLAG` interface.
- `local_alg` : algorithm of optimization. Derivative-free and gradient-based algorithms form NLopt package. 
- `max_iter` : maximal number of fitter iterations. If reaches the result status will be `:MAX_ITER_STOP`.
- `loss_grad` : For gradient optimization methods it is necessary to set how the gradient of `loss_func` should be calculated.
    There are options:
    - `:EMPTY` (default) no gradient is set. It works only for gradient-free methods.
    - `:AUTODIFF` means autodifferentiation from `ForwardDiff` package is used.
    - `:FINITE` means finite difference method from `Calculus` is used.
    - It is also possible to set gradient function here `function(x::Vector{Float64})` which returns gradient vector.
- `silent` : Boolean argument declaring whether we display the optimization progress. Default is `false`

See also [`get_interval`](@ref)
"""
function get_endpoint(
    theta_init::Vector{Float64},
    theta_num::Int,
    loss_func::Function,
    method::Symbol,
    direction::Symbol = :right;

    loss_crit::Float64 = 0.0,
    scale::Vector{Symbol} = fill(:direct, length(theta_init)), # :direct, :lin, :log, :logit
    theta_bounds::Vector{Tuple{Float64,Float64}} = unscaling.(
        fill(DEFAUL_THETA_BOUNDS_ITEM, length(theta_init)),
        scale
        ),
    scan_bound::Float64 = unscaling(
        (direction==:left) ? -9.0 : 9.0,
        scale[theta_num]
        ),
    scan_tol::Float64 = 1e-3,
    loss_tol::Float64 = DEFAULT_LOSS_TOL,
    local_alg::Symbol = DEFAULT_LOCAL_ALG,
    max_iter::Int = 10^5,
    loss_grad::Union{Function, Symbol} = :EMPTY,
    silent::Bool = false,
    #kwargs... # other options for get_right_endpoint
)
    isLeft = direction == :left
    n_theta = length(theta_init)

    # checking arguments
    # theta_bound[1] < theta_init < theta_bound[2]
    theta_init_outside_theta_bounds = .! [theta_bounds[i][1] < theta_init[i] < theta_bounds[i][2] for i in 1:length(theta_init)]
    if any(theta_init_outside_theta_bounds)
        throw(ArgumentError("theta_init is outside theta_bound: $(findall(theta_init_outside_theta_bounds))"))
    end
    # scan_bound should be within theta_bounds
    !(theta_bounds[theta_num][1] < scan_bound < theta_bounds[theta_num][2]) &&
        throw(ArgumentError("scan_bound are outside of the theta_bounds $(theta_bounds[theta_num])"))
    # theta_init should be within scan_bound
    if (theta_init[theta_num] >= scan_bound && !isLeft) || (theta_init[theta_num] <= scan_bound && isLeft)
        throw(ArgumentError("init values are outside of the scan_bound $scan_bound"))
    end
    # 0 <= theta_bound[1] for :log
    less_than_zero_theta_bounds = (scale .== :log) .& [theta_bounds[i][1] < 0 for i in 1:length(theta_init)]
    if any(less_than_zero_theta_bounds)
        throw(ArgumentError(":log scaled theta_bound min is negative: $(findall(less_than_zero_theta_bounds))"))
    end
    # 0 <= theta_bounds <= 1 for :logit
    less_than_zero_theta_bounds = (scale .== :logit) .& [theta_bounds[i][1] < 0 || theta_bounds[i][2] > 1 for i in 1:length(theta_init)]
    if any(less_than_zero_theta_bounds)
        throw(ArgumentError(":logit scaled theta_bound min is outside range [0,1]: $(findall(less_than_zero_theta_bounds))"))
    end
    # loss_func(theta_init) < loss_crit
    !(loss_func(theta_init) < loss_crit) &&
        throw(ArgumentError("Check theta_init and loss_crit: loss_func(theta_init) should be < loss_crit"))

    # set counter in the scope
    counter::Int = 0
    # set supreme, maximal or minimal value of scanned parameter inside critical
    supreme_gd = nothing

    # progress info
    prog = ProgressUnknown(; desc = "$direction CP counter:", spinner=false, enabled=!silent, showspeed=true)

    # transforming loss
    theta_init_gd = scaling.(theta_init, scale)
    if isLeft theta_init_gd[theta_num] *= -1 end # change direction

    function loss_func_gd(theta_gd)
        theta_g = copy(theta_gd)
        if isLeft theta_g[theta_num] *= -1 end # change direction
        theta = unscaling.(theta_g, scale)
        # calculate function
        loss_norm = loss_func(theta) - loss_crit

        # update counter
        counter += 1
        # update supreme ?
        update_supreme = (loss_norm < 0.) &&
            (typeof(supreme_gd)==Nothing || (theta_gd[theta_num] > supreme_gd))
        if update_supreme
            supreme_gd = theta_gd[theta_num]
        end
        # display current
        supreme = if (isLeft && typeof(supreme_gd)!==Nothing)
            unscaling(-supreme_gd, scale[theta_num])
        else
            unscaling(supreme_gd, scale[theta_num])
        end
        ProgressMeter.update!(prog, counter, spinner="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"; showvalues = [(:supreme,round(supreme; sigdigits=4))])

        return loss_norm
    end
    theta_bounds_gd = scaling.(theta_bounds, scale)
    if isLeft theta_bounds_gd[theta_num] = (-1*theta_bounds_gd[theta_num][2], -1*theta_bounds_gd[theta_num][1]) end # change direction

    scan_bound_gd = scaling(scan_bound, scale[theta_num])
    if isLeft scan_bound_gd *= -1 end # change direction

    # transform gradient
    loss_grad_gd = loss_grad
    loss_grad_gd = if isa(loss_grad, Function)
        function(theta_gd)
            theta_g = copy(theta_gd)
            if isLeft theta_g[theta_num] *= -1 end # change direction
            theta = unscaling.(theta_g, scale)
            loss_grad_value = loss_grad(theta)
            loss_grad_value_gd = zeros(n_theta)

            for i in 1:n_theta
                if scale[i] == :log
                    loss_grad_value_gd[i] = loss_grad_value[i] * theta[i] * log(10.)
                elseif scale[i] == :logit
                    loss_grad_value_gd[i] = loss_grad_value[i] * theta[i] * (1. - theta[i]) * log(10.)
                else # for :direct and :lin
                    loss_grad_value_gd[i] = loss_grad_value[i]
                end
            end

            if isLeft loss_grad_value_gd[theta_num] *= -1 end # change direction

            return loss_grad_value_gd
        end
    else
        loss_grad
    end

    # calculate endpoint using base method
    (optf_gd, pp_gd, status) = get_right_endpoint(
        theta_init_gd,
        theta_num,
        loss_func_gd,
        Val(method);
        theta_bounds = theta_bounds_gd,
        scan_bound = scan_bound_gd,
        scan_tol,
        loss_tol,
        local_alg,
        max_iter,
        loss_grad = loss_grad_gd
        #kwargs...
    )

    # transforming back
    if (isLeft && typeof(optf_gd)!==Nothing) optf_gd *= -1 end # change direction
    optf = unscaling(optf_gd, scale[theta_num])
    temp_fun = (pp::ProfilePoint) -> begin
        if isLeft pp.params[theta_num] *= -1 end # change direction
        ProfilePoint(
            unscaling(pp.params[theta_num], scale[theta_num]),
            pp.loss + loss_crit,
            unscaling.(pp.params, scale),
            pp.ret,
            pp.counter
        )
    end
    pps = [ temp_fun(pp_gd[i]) for i in 1:length(pp_gd) ]
    # transforming supreme back
    supreme = if (isLeft && typeof(supreme_gd)!==Nothing) 
        unscaling(-supreme_gd, scale[theta_num])
    else
        unscaling(supreme_gd, scale[theta_num])
    end
    ProgressMeter.finish!(prog)

    EndPoint(optf, pps, status, direction, counter, supreme)
end

"""
    function get_endpoint(
        theta_init::Vector{Float64},
        scan_func::Function,
        loss_func::Function,
        method::Symbol,
        direction::Symbol = :right;

        loss_crit::Float64 = 0.0,
        scale::Vector{Symbol} = fill(:direct, length(theta_init)),
        theta_bounds::Vector{Tuple{Float64,Float64}} = unscaling.(
            fill((-Inf, Inf), length(theta_init)),
            scale
            ),
        scan_bound::Float64 = unscaling(
            (direction==:left) ? -9.0 : 9.0,
            scale[theta_num]
            ),
        scan_tol::Float64 = 1e-3,
        loss_tol::Float64 = 0.,
        local_alg::Symbol = :LN_NELDERMEAD,
        kwargs...
        )

Calculates confidence interval's right or left endpoints for a function of parameters `scan_func`.

## Return
[`EndPoint`](@ref) object storing confidence interval's endpoint and intermediate profile points.

## Arguments
- `theta_init`: starting values of parameter vector ``\\theta``. The starting values should not necessary be the optimum values of `loss_func` but `loss_func(theta_init)` should be lower than `loss_crit`.
- `scan_func`: function of parameters.
- `loss_func`: loss function ``\\Lambda\\left(\\theta\\right)`` for profile likelihood-based (PL) identification. Usually we use log-likelihood for PL analysis: ``\\Lambda( \\theta ) = - 2 ln\\left( L(\\theta) \\right)``.
- `method`: computational method to estimate confidence interval's endpoint. Currently the only supported method is: `:CICO_ONE_PASS`.
- `direction`: `:right` or `:left` endpoint to estimate.

## Keyword arguments
- `loss_crit`: critical level of loss function. Confidence interval's endpoint value is the intersection point of profile likelihood and `loss_crit` level.
- `scale`: vector of scale transformations for each parameters' component. Possible values: `:direct` (`:lin`), `:log`, `:logit`. This option can speed up the optimization, especially for wide `theta_bounds`. The default value is `:direct` (no transformation) for all parameters.
- `theta_bounds`: vector of tuple `(lower_bound, upper_bound)` for each parameter. Bounds define the ranges for possible parameter values. Default bounds are `(-Inf,Inf)`.
- `scan_bound`: value which states the area of confidence point analysis.
- `scan_tol`: Absolute tolerance for `theta_num` parameter used as termination criterion.  
- `loss_tol`: Absolute tolerance controlling `loss_func` closeness to `loss_crit` (termination criterion). Currently doesn't work for `:CICO_ONE_PASS` method because of limitation in `LN_AUGLAG` interface.
- `local_alg`: algorithm of optimization. Derivative-free and gradient-based algorithms form NLopt package. 
- `max_iter` : maximal number of fitter iterations. If reaches the result status will be `:MAX_ITER_STOP`.
- `scan_grad` : For gradient optimization methods it is necessary to set how the gradient of `scan_func` should be calculated.
    - `:EMPTY` (default) no gradient is set. It works only for gradient-free methods.
    - `:AUTODIFF` means autodifferentiation from `ForwardDiff` package is used.
    - `:FINITE` means finite difference method from `Calculus` is used.
    - `function(x::Vector{Float64})` which returns gradient vector.
- `loss_grad` : For gradient optimization methods it is necessary to set how the gradient of `loss_func` should be calculated.
    - `:EMPTY` (default) no gradient is set. It works only for gradient-free methods.
    - `:AUTODIFF` means autodifferentiation from `ForwardDiff` package is used.
    - `:FINITE` means finite difference method from `Calculus` is used.
    - `function(x::Vector{Float64})` which returns gradient vector.
- `silent` : Boolean argument declaring whether we display the optimization progress. Default is `false`

See also [`get_interval`](@ref)
"""
function get_endpoint(
    theta_init::Vector{Float64},
    scan_func::Function,
    loss_func::Function,
    method::Symbol,
    direction::Symbol = :right;

    loss_crit::Float64 = 0.0,
    # :direct, :lin, :log, :logit
    scale::Vector{Symbol} = fill(:direct, length(theta_init)),
    theta_bounds::Vector{Tuple{Float64,Float64}} = unscaling.(
        fill(DEFAUL_THETA_BOUNDS_ITEM, length(theta_init)),
        scale
        ),
    scan_bound::Float64 = (direction==:left) ? -1e9 : 1e9, # log scan bound is not implemented
    scan_tol::Float64 = 1e-3,
    loss_tol::Float64 = DEFAULT_LOSS_TOL,
    local_alg::Symbol = DEFAULT_LOCAL_ALG,
    max_iter::Int = 10^5,
    scan_grad::Union{Function, Symbol} = :EMPTY,
    loss_grad::Union{Function, Symbol} = :EMPTY,
    silent::Bool = false,
    #kwargs... # other options for get_right_endpoint
)
    isLeft = direction == :left
    n_theta = length(theta_init)
    loss_init = loss_func(theta_init)

    # checking arguments
    # theta_bound[1] < theta_init < theta_bound[2]
    theta_init_outside_theta_bounds = .! [theta_bounds[i][1] < theta_init[i] < theta_bounds[i][2] for i in 1:length(theta_init)]
    if any(theta_init_outside_theta_bounds)
        throw(ArgumentError("theta_init is outside theta_bound: $(findall(theta_init_outside_theta_bounds))"))
    end

    # 0 <= theta_bound[1] for :log
    less_than_zero_theta_bounds = (scale .== :log) .& [theta_bounds[i][1] < 0 for i in 1:length(theta_init)]
    if any(less_than_zero_theta_bounds)
        throw(ArgumentError(":log scaled theta_bound min is negative: $(findall(less_than_zero_theta_bounds))"))
    end
    # 0 <= theta_bounds <= 1 for :logit
    less_than_zero_theta_bounds = (scale .== :logit) .& [theta_bounds[i][1] < 0 || theta_bounds[i][2] > 1 for i in 1:length(theta_init)]
    if any(less_than_zero_theta_bounds)
        throw(ArgumentError(":logit scaled theta_bound min is outside range [0,1]: $(findall(less_than_zero_theta_bounds))"))
    end
    # loss_func(theta_init) < loss_crit
    !(loss_init < loss_crit) &&
        throw(ArgumentError("Check theta_init and loss_crit: loss_func(theta_init) should be < loss_crit"))

    # set counter in the scope
    prog = ProgressUnknown(; desc = "Fitter counter:", spinner=false, enabled=!silent, showspeed=true)
    counter::Int = 0
    # set supreme, maximal or minimal value of loss_func inside critical
    supreme_gd = nothing
    scan_val_gd = nothing

    # transforming
    theta_init_gd = scaling.(theta_init, scale)
    
    function scan_func_gd(theta_gd)
        theta = unscaling.(theta_gd, scale)
        scan_val = scan_func(theta)
        scan_val_gd = isLeft ? (-1)*scan_val : scan_val
        
        return scan_val_gd
    end

    function loss_func_gd(theta_gd)
        #theta_g = copy(theta_gd) # why copy?
        theta = unscaling.(theta_gd, scale)
        loss_value = loss_func(theta) - loss_crit

        counter += 1
        #println("$loss_value => $supreme_gd")
        should_update = (loss_value < 0.) &&
            !isa(scan_val_gd, Nothing) &&
            (isa(supreme_gd, Nothing) || (scan_val_gd > supreme_gd)) &&
            !isa(scan_val_gd, ForwardDiff.Dual)
        if should_update
            supreme_gd = scan_val_gd
        end
        supreme = if isa(supreme_gd, Nothing)
            "-"
        elseif isLeft
            round(-supreme_gd; sigdigits=4)
        else
            round(supreme_gd; sigdigits=4)
        end
        ProgressMeter.update!(prog, counter; showvalues = [(:supreme,supreme)])

        return loss_value
    end
    theta_bounds_gd = scaling.(theta_bounds, scale)

    # TODO: transformed by scan_scale: scaling(scan_bound, scan_scale)
    scan_bound_gd = scan_bound
    if isLeft scan_bound_gd *= -1 end # change direction

    # transform gradients
    scan_grad_gd = if isa(scan_grad, Function)
        function(theta_gd)
            theta = unscaling.(theta_gd, scale)
            scan_grad_value = scan_grad(theta)
            scan_grad_value_gd = zeros(n_theta)

            for i in 1:n_theta
                if scale[i] == :log
                    scan_grad_value_gd[i] = scan_grad_value[i] * theta[i] * log(10.)
                elseif scale[i] == :logit
                    scan_grad_value_gd[i] = scan_grad_value[i] * theta[i] * (1. - theta[i]) * log(10.)
                else # for :direct and :lin
                    scan_grad_value_gd[i] = scan_grad_value[i]
                end
            end

            return scan_grad_value_gd
        end
    else
        scan_grad
    end
    
    loss_grad_gd = if isa(loss_grad, Function)
        function(theta_gd)
            theta = unscaling.(theta_gd, scale)
            loss_grad_value = loss_grad(theta)
            loss_grad_value_gd = zeros(n_theta)

            for i in 1:n_theta
                if scale[i] == :log
                    loss_grad_value_gd[i] = loss_grad_value[i] * theta[i] * log(10.)
                elseif scale[i] == :logit
                    loss_grad_value_gd[i] = loss_grad_value[i] * theta[i] * (1. - theta[i]) * log(10.)
                else # for :direct and :lin
                    loss_grad_value_gd[i] = loss_grad_value[i]
                end
            end

            return loss_grad_value_gd
        end
    else
        loss_grad
    end

    # calculate endpoint using base method
    (optf_gd, pp_gd, status) = get_right_endpoint(
        theta_init_gd,
        scan_func_gd,
        loss_func_gd,
        Val(method);

        theta_bounds = theta_bounds_gd,
        scan_bound = scan_bound_gd,
        scan_tol,
        loss_tol,
        local_alg,
        max_iter,
        scan_grad = scan_grad_gd,
        loss_grad = loss_grad_gd,
        #kwargs...
    )

    # transforming back
    temp_fun = (pp::ProfilePoint) -> begin
        value = isLeft ? (-1)*pp.value : pp.value # TODO: scan_scale
        ProfilePoint(
            value,
            pp.loss + loss_crit,
            unscaling.(pp.params, scale),
            pp.ret,
            pp.counter
        )
    end
    pps = [ temp_fun(pp_gd[i]) for i in 1:length(pp_gd) ]
    # transforming supreme back
    if (isLeft && typeof(supreme_gd)!==Nothing) supreme_gd *= -1 end # change direction

    if (isLeft && typeof(optf_gd)!==Nothing) optf_gd *= -1 end # change direction
    # optf = unscaling(optf_gd, scan_scale)
    optf = optf_gd

    EndPoint(optf, pps, status, direction, counter, supreme_gd)
end
