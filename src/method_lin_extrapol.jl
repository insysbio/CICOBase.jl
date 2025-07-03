function get_right_endpoint(
    theta_init::Vector{Float64}, # initial point of parameters
    theta_num::Int, # number of parameter to scan
    loss_func::Function, # lambda(theta) - labmbda_min - delta_lambda
    method::Val{:LIN_EXTRAPOL}; # function works only for method LIN_INTER

    theta_bounds::Vector{Tuple{Float64,Float64}} = fill(DEFAUL_THETA_BOUNDS_ITEM, length(theta_init)),
    scan_bound::Float64 = 9.0,
    scan_tol::Float64 = 1e-3,
    loss_tol::Float64 = DEFAULT_LOSS_TOL,
    # method args
    scan_hini = 1.,
    scan_hmax = Inf,
    # local alg args
    local_alg::Symbol = DEFAULT_LOCAL_ALG,
    max_iter::Int = 10^5,
    kwargs... # options for local fitter
    )
    # dim of the theta vector
    n_theta = length(theta_init)

    # checking arguments
    # methods which are not supported
    if local_alg in [:LN_BOBYQA, :LN_SBPLX, :LN_NEWUOA]
        @warn "Using local_alg = :"*String(local_alg)*" may result in wrong output."
    end

    # to count loss function calls inside this function, accumulation
    accum_counter::Int = 0
    # empty container
    pps = ProfilePoint[]

    prof = _profile(
        theta_init,
        theta_num,
        loss_func;
        theta_bounds = theta_bounds,
        local_alg = local_alg,
        ftol_abs = loss_tol
    )

    # first iteration
    x_2 = theta_init[theta_num]
    theta_init_2 = theta_init
    iteration_count = 0

    # other iterations
    while true
        # preparation
        global x_1 # not initialized for the first iteration
        global point_1 # not initialized for the first iteration
        iteration_count += 1 # to understand if this is a first iteration

        # get profile point
        point_2 = prof(
            x_2;
            theta_init_i = theta_init_2, # hypothetically this makes optimization more effective
            maxeval = max_iter - accum_counter # how many calls left
            )
        push!(pps, point_2)
        accum_counter += point_2.counter # update counter

        if point_2.ret == :MAXEVAL_REACHED
            return (nothing, pps, :MAX_ITER_STOP)
        elseif point_2.ret == :FORCED_STOP
            return (nothing, pps, :LOSS_ERROR_STOP)
        elseif x_2 >= scan_bound && point_2.loss < 0. # successful result
            return (nothing, pps, :SCAN_BOUND_REACHED)
        # no checking for the first iteration
        elseif iteration_count>1 && isapprox((x_2 - x_1) * point_2.loss / (point_2.loss - point_1.loss), 0., atol = scan_tol) # successful result
            return (x_2, pps, :BORDER_FOUND_BY_SCAN_TOL)
        elseif isapprox(point_2.loss, 0., atol = loss_tol) # successful result
            return (x_2, pps, :BORDER_FOUND_BY_LOSS_TOL)
        end

        # next step
        if iteration_count==1
            x_3_extrapol = x_2 + scan_hini
        else
            if (point_2.loss - point_1.loss) / (x_2 - x_1) <= 0.
                x_3_extrapol = x_2 + scan_hini
            else
                x_3_extrapol = x_2 + (x_2 - x_1) * (0. - point_2.loss) / (point_2.loss - point_1.loss)
            end
        end
        x_3 = minimum([x_2+scan_hmax, x_3_extrapol, theta_bounds[theta_num][2]])
        # preparation for the next iteration
        (point_1, x_1, x_2, theta_init_2) = (point_2, x_2, x_3, point_2.params)
    end
end
