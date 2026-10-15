@testset "default options" begin
    res0 = [get_interval(
        [3., 2., 2.1],
        i,
        (x::Vector{Float64}) -> f_3p_1im_dep(x),
        :CICO_ONE_PASS;
        loss_crit = 9.,
        silent = true
    ) for i in 1:3]

    @test isapprox(res0[1].result[1].value, 1.0, atol=1e-2)
    @test isapprox(res0[1].result[2].value, 5.0, atol=1e-2)
    @test length(res0[1].result[1].profilePoints) > 0
    @test length(res0[1].result[2].profilePoints) > 0
    @test res0[1].result[1].status == :BORDER_FOUND_BY_SCAN_TOL
    @test res0[1].result[2].status == :BORDER_FOUND_BY_SCAN_TOL
    @test res0[1].result[1].direction == :left
    @test res0[1].result[2].direction == :right
    @test isapprox(res0[2].result[1].value, 2.0-2.0*sqrt(2.), atol=1e-2)
    @test isapprox(res0[2].result[2].value, 2.0+2.0*sqrt(2.), atol=1e-2)
    @test res0[2].result[1].status == :BORDER_FOUND_BY_SCAN_TOL
    @test res0[2].result[2].status == :BORDER_FOUND_BY_SCAN_TOL
    @test res0[3].result[1].status == :SCAN_BOUND_REACHED
    @test res0[3].result[2].status == :SCAN_BOUND_REACHED

    options = res0[1].input.options
    @test options[:loss_crit] == 9.0
    @test options[:scan_bounds] == (-9.0, 9.0)
    @test options[:scale] == [:direct, :direct, :direct]
    @test options[:silent] == true
end

@testset "parameter scan with log scale" begin
    pi = get_interval(
        [3.0, 2.0, 2.1], 1, f_3p_1im_dep, :CICO_ONE_PASS;
        loss_crit = 9.0, scale = fill(:log, 3), silent = true
    )

    @test all(ep -> ep.status == :BORDER_FOUND_BY_SCAN_TOL, pi.result)
    @test isapprox(log10(pi.result[1].value), log10(1.0); atol = 1e-2)
    @test isapprox(log10(pi.result[2].value), log10(5.0); atol = 1e-2)
end

@testset "function scan default bounds" begin
    center = [2.0, 0.5]
    loss(p) = sum(abs2, p .- center)
    scan(p) = p[1] - 4 * p[2]
    radius = sqrt(0.17)
    for scale in ([:direct, :direct], [:log, :logit])
        pi = get_interval(center, scan, loss, :CICO_ONE_PASS;
            scale, loss_crit = 0.01, local_alg = :LD_SLSQP,
            scan_tol = 1e-7, scan_grad = :AUTODIFF,
            loss_grad = :AUTODIFF, silent = true
        )
        @test pi.input.options[:scan_bounds] == (-9.0, 9.0)
        @test isapprox(pi.result[1].value, -radius; atol = 1e-3)
        @test isapprox(pi.result[2].value, radius; atol = 1e-3)
        @test all(ep -> ep.status == :BORDER_FOUND_BY_SCAN_TOL, pi.result)
        @test all(pp -> isapprox(pp.value, scan(pp.params); atol = 1e-9),
            [pp for ep in pi.result for pp in ep.profilePoints])
    end
end
