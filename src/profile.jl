
# Internal routine: inputs, callback arguments, and results share the caller's
# coordinates. No scale conversions are performed here.
function _profile(
    theta_init::Vector{Float64},
    theta_num::Int,
    loss_func::Function;

    skip_optim::Bool = false,
    theta_bounds::Vector{Tuple{Float64,Float64}} = fill(DEFAUL_THETA_BOUNDS_ITEM, length(theta_init)),
    # fit alg args
    local_alg::Symbol = DEFAULT_LOCAL_ALG,
    local_initial_step = theta_init,
    ftol_abs::Float64 = 1e-3,
    maxeval::Int = 10^5,
    kwargs...
)
    theta_length = length(theta_init)
    step = _validated_local_initial_step(local_initial_step, theta_length)

    # remove target parameter from the list
    indexes_rest = [i for i in 1:theta_length]
    deleteat!(indexes_rest, theta_num)
    # set bounds
    lb = [theta_bounds[i][1] for i in indexes_rest]
    ub = [theta_bounds[i][2] for i in indexes_rest]

    if skip_optim || theta_length == 1 # if profile == loss_func
        return (x::Float64; theta_init_i::Vector{Float64} = theta_init, maxeval::Int = maxeval) -> begin
            theta_full = copy(theta_init_i)
            splice!(theta_full, theta_num, x)
            loss = loss_func(theta_full)

            # return
            ProfilePoint(
                x,
                loss,
                theta_full,
                :OPTIMIZATION_SKIPPED,
                1 # counter, only one call
            )
        end
    else
        # set optimizer
        opt = Opt(local_alg, theta_length - 1)
        ftol_abs!(opt, ftol_abs)
        NLopt.nlopt_set_param(opt, "tolg", MAGIC_TOLG) # default 1e-8
        #xtol_abs!(opt, 0.)
        lower_bounds!(opt, lb)
        upper_bounds!(opt, ub)
        # initial step for optimization
        initial_step!(opt, step isa AbstractVector ? step[indexes_rest] : step)

        # profile function
        return (x::Float64; theta_init_i::Vector{Float64} = theta_init, maxeval::Int = maxeval) -> begin
            # to count loss function calls inside profile
            counter::Int = 0
            # get init of rest component
            theta_init_rest = copy(theta_init_i)
            deleteat!(theta_init_rest, theta_num)
            # get rest of loss_func
            loss_func_rest = (theta_rest::Array{Float64, 1}, g::Array{Float64, 1}) -> begin
                theta_full = copy(theta_rest)
                splice!(theta_full, theta_num:(theta_num-1), x)
                loss = try
                    loss_func(theta_full)
                catch e
                    @warn "Error when call loss_func($theta_full)"
                    #throw(e)
                    throw(NLopt.ForcedStop())
                end
                counter += 1 # update counter

                return loss
            end
            # set optimizer
            min_objective!(opt, loss_func_rest)
            maxeval!(opt, maxeval)
            # start optimization
            (loss, theta_opt, ret) = optimize(opt, theta_init_rest)
            splice!(theta_opt, theta_num:(theta_num-1), x)
            # return
            ProfilePoint(
                x,
                loss,
                theta_opt,
                ret,
                counter
            )
        end
    end
end

"""
    function profile(
        theta_init::Vector{Float64},
        theta_num::Int,
        loss_func::Function;

        skip_optim::Bool = false,
        scale::Vector{Symbol} = fill(:direct, length(theta_init)),
        theta_bounds::Vector{Tuple{Float64,Float64}} = unscaling.(
            fill((-Inf, Inf), length(theta_init)),
            scale
            ),
        local_alg::Symbol = :LN_NELDERMEAD,
        local_initial_step = nothing,
        ftol_abs::Float64 = 1e-3,
        maxeval::Int = 10^5,
        kwargs... # currently not used
    )
Generates a profile function based on `loss_func`. Extrapolation methods use `_profile` directly.

## Return
Returns a profile function for the selected parameter. Constructor inputs and
`loss_func` arguments use original units. Returned `ProfilePoint.params` and
the optional `theta_init_i` use internal coordinates: the scanned component is
direct, and the others follow `scale`. A previous point's `params` can be reused
as `theta_init_i`.

# Arguments
- `theta_init`: starting values of parameter vector ``\\theta``. 
- `theta_num`: index of vector component for identification: `theta_init(theta_num)`.
- `loss_func`: loss function ``\\Lambda\\left(\\theta\\right)`` for profile likelihood-based (PL) identification. Usually we use log-likelihood for PL analysis: ``\\Lambda( \\theta ) = - 2 ln\\left( L(\\theta) \\right)``.

## Keyword arguments
- `skip_optim` : set `true` if you need marginal profile, i.e. profile without optimization. Default is `false`.
- `scale`: per-parameter internal transformations; the scanned component remains direct.
- `theta_bounds`: vector of tuple `(lower_bound, upper_bound)` for each parameter. Bounds define the ranges for possible parameter values. Default bounds are `(-Inf,Inf)`.
- `local_alg`: algorithm of optimization. Derivative-free and gradient-based algorithms form NLopt package.
- `local_initial_step`: nonzero scalar or full parameter vector of initial steps in the optimization coordinates; the scanned component is ignored. By default, uses the transformed starting values; a zero component raises an error.
- `ftol_abs` : absolute tolerance criterion for profile function.
- `maxeval` : maximal number of `loss_func` evaluations to estimate profile point.
"""
function profile(
    theta_init::Vector{Float64},
    theta_num::Int,
    loss_func::Function;

    skip_optim::Bool = false,
    scale::Vector{Symbol} = fill(:direct, length(theta_init)),
    theta_bounds::Vector{Tuple{Float64,Float64}} = unscaling.(
        fill(DEFAUL_THETA_BOUNDS_ITEM, length(theta_init)),
        scale
        ),
    # fit alg args
    local_alg::Symbol = DEFAULT_LOCAL_ALG,
    local_initial_step = nothing,
    ftol_abs::Float64 = 1e-3,
    maxeval::Int = 10^5,
    kwargs...
)
    scale_mod = copy(scale)
    scale_mod[theta_num] = :direct

    theta_init_g = scaling.(theta_init, scale_mod)
    theta_bounds_g = scaling.(theta_bounds, scale_mod)

    function loss_func_g(theta_g::Vector{Float64})
        theta = unscaling.(theta_g, scale_mod)
        return loss_func(theta)
    end

    return _profile(
        theta_init_g,
        theta_num,
        loss_func_g;
        skip_optim = skip_optim,
        theta_bounds = theta_bounds_g,
        local_alg = local_alg,
        local_initial_step = isnothing(local_initial_step) ? theta_init_g : local_initial_step,
        ftol_abs = ftol_abs,
        maxeval = maxeval,
        kwargs...
    )
end
