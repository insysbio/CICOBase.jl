@testset "profile" begin
    prof = CICOBase._profile(
        [3., 2., 2.1],
        1,
        f_3p_1im_dep;
        skip_optim = false
    )

    x =  0.:0.2:10
    y = [prof(x[i]) for i in 1:length(x)]
    
    @test isapprox(y[5].value, 0.8)
    @test isapprox(y[5].loss, 9.8402, atol=1e-3)
    @test y[5].ret == :FTOL_REACHED
    @test y[5].counter > 1
    @test length(y[5].params) == 3
end

@testset "update_profile_points! with scale" begin
    # Regression: refinement must not minimize over negative log-scaled parameters.
    loss(p) = p[1]^2 + (p[2] + 1)^2
    pi = get_interval(
        [0.0, 1.0], 1, loss, :LIN_EXTRAPOL;
        scale = [:direct, :log], loss_crit = 9.0, loss_tol = 1e-5, silent = true
    )
    counts = [length(ep.profilePoints) for ep in pi.result]
    update_profile_points!(pi; max_recursions = 0)
    for (ep, n) in zip(pi.result, counts)
        added = ep.profilePoints[n+1:end]
        @test !isempty(added)
        @test all(pp -> pp.params[2] >= 0.0, added)
        @test all(pp -> isapprox(pp.loss, pp.value^2 + 1.0; atol = 1e-4), added)
    end

    # Check default/explicit bounds and storage in original coordinates.
    for s in (:direct, :lin, :log, :logit), explicit_bounds in (false, true)
        loss_i(p) = begin
            @assert s != :log || p[2] > 0.0
            @assert s != :logit || 0.0 < p[2] < 1.0
            (p[1] - 0.5)^2 + (p[2] - 0.7)^2
        end
        options = Dict{Symbol,Any}(:scale => [s, s], :loss_tol => 1e-8)
        if explicit_bounds
            options[:theta_bounds] = [(0.1, 0.9), (0.1, 0.9)]
        end
        input = ParamIntervalInput([0.5, 0.3], 1, loss_i, :LIN_EXTRAPOL, options)
        endpoints = (
            EndPoint(0.2, ProfilePoint[], :BORDER_FOUND_BY_SCAN_TOL, :left, 0, 0.2),
            EndPoint(0.8, ProfilePoint[], :BORDER_FOUND_BY_SCAN_TOL, :right, 0, 0.8)
        )
        interval = ParamInterval(input, loss_i(input.theta_init), :LIN_EXTRAPOL, endpoints)
        update_profile_points!(interval; max_recursions = 0)
        for ep in interval.result
            @test !isempty(ep.profilePoints)
            @test all(pp -> pp.params[1] == pp.value, ep.profilePoints)
            @test all(pp -> isapprox(pp.params[2], 0.7; atol = 1e-3), ep.profilePoints)
            @test all(pp -> isapprox(pp.loss, (pp.value - 0.5)^2; atol = 1e-6), ep.profilePoints)
            @test all(pp -> pp.loss == loss_i(pp.params), ep.profilePoints)
        end
    end
end

@testset "update_profile_points! skips missing endpoints" begin
    seed = ProfilePoint(0.0, 0.0, [0.0, 0.0], :FTOL_REACHED, 1)
    for status in (:SCAN_BOUND_REACHED, :MAX_ITER_STOP, :FORCED_STOP,
                   :OPTIMIZATION_EXCEPTION_STOP, :UNKNOWN_STOP)
        calls = Ref(0)
        loss(p) = begin
            calls[] += 1
            sum(abs2, p)
        end
        input = ParamIntervalInput([0.0, 0.0], 1, loss, :CICO_ONE_PASS, Dict{Symbol,Any}())
        endpoints = (
            EndPoint(nothing, [seed], status, :left, 1, nothing),
            EndPoint(nothing, [seed], status, :right, 1, nothing)
        )
        pi = ParamInterval(input, 0.0, :CICO_ONE_PASS, endpoints)
        @test update_profile_points!(pi; max_recursions = 0) === nothing
        @test calls[] == 0
        @test all(ep -> ep.profilePoints == [seed], pi.result)
    end

    # A missing endpoint must not prevent refinement of the other side.
    finite_loss(p) = sum(abs2, p)
    input = ParamIntervalInput(
        [0.0, 0.0], 1, finite_loss, :LIN_EXTRAPOL, Dict{Symbol,Any}(:loss_tol => 1e-8)
    )
    missing = EndPoint(nothing, [seed], :MAX_ITER_STOP, :left, 1, nothing)
    finite = EndPoint(1.0, ProfilePoint[], :BORDER_FOUND_BY_SCAN_TOL, :right, 1, 1.0)
    pi = ParamInterval(input, 0.0, :LIN_EXTRAPOL, (missing, finite))
    @test update_profile_points!(pi; max_recursions = 0) === nothing
    @test missing.profilePoints == [seed]
    @test !isempty(finite.profilePoints)
    @test all(pp -> isapprox(pp.loss, finite_loss(pp.params)), finite.profilePoints)
end

@testset "skipped profile uses supplied parameters" begin
    initial = [1.0, 2.0, 3.0]
    warm = [10.0, 20.0, 30.0]
    prof = CICOBase._profile(initial, 2, p -> sum(abs2, p); skip_optim = true)
    pp = prof(4.0; theta_init_i = warm)
    @test pp.value == 4.0
    @test pp.params == [10.0, 4.0, 30.0]
    @test pp.loss == 1016.0
    @test pp.ret == :OPTIMIZATION_SKIPPED
    @test pp.counter == 1
    @test warm == [10.0, 20.0, 30.0]
    @test initial == [1.0, 2.0, 3.0]
    @test prof(4.0).params == [1.0, 4.0, 3.0]

    # profile accepts warm starts and returns params in internal coordinates.
    # The scanned component remains direct; the loss callback receives original values.
    seen = Vector{Float64}[]
    loss(p) = begin
        push!(seen, copy(p))
        sum(abs2, p)
    end
    scale = [:log, :log, :logit]
    scale_mod = [:log, :direct, :logit]
    prof_scaled = CICOBase.profile([2.0, 10.0, 0.4], 2, loss; scale, skip_optim = true)
    warm_scaled = scaling.([20.0, 30.0, 0.6], scale_mod)
    before = copy(warm_scaled)
    pp_scaled = prof_scaled(15.0; theta_init_i = warm_scaled)
    @test isapprox(pp_scaled.params, scaling.([20.0, 15.0, 0.6], scale_mod))
    @test isapprox(pp_scaled.loss, 625.36)
    @test isapprox(only(seen), [20.0, 15.0, 0.6])
    @test pp_scaled.counter == 1
    @test warm_scaled == before

    default = prof_scaled(15.0)
    @test isapprox(default.params, scaling.([2.0, 15.0, 0.4], scale_mod))
    @test isapprox(seen[end], [2.0, 15.0, 0.4])
end
