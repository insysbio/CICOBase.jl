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
