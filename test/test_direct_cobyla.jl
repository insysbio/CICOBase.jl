@testset "DIRECT_COBYLA" begin
    loss(x) = (x[1] - 3.0)^2
    interval = get_interval([3.0], 1, loss, :DIRECT_COBYLA;
        loss_crit=4.0, loss_tol=1e-8, scan_tol=1e-8, local_initial_step=0.5, silent=true)
    @test all(p -> p.status == :BORDER_FOUND_BY_SCAN_TOL, interval.result)
    @test isapprox(interval.result[1].value, 1.0; atol=1e-4)
    @test isapprox(interval.result[2].value, 5.0; atol=1e-4)
    scan(x) = x[1]
    endpoint = get_endpoint([3.0], scan, loss, :DIRECT_COBYLA;
        loss_crit=4.0, loss_tol=1e-8, scan_tol=1e-8, local_initial_step=0.5, silent=true)
    @test endpoint.status == :BORDER_FOUND_BY_SCAN_TOL
    @test isapprox(endpoint.value, 5.0; atol=1e-4)
end
