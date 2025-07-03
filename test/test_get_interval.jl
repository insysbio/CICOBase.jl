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
