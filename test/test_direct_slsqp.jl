@testset "DIRECT_SLSQP with automatic gradients" begin
    test_alg_interval((algorithm=:LD_SLSQP, skip=[]);
        method=:DIRECT_SLSQP, bounds=(-1e10, 1e10),
        loss_grad=:AUTODIFF, loss_tol=1e-8)
end

@testset "one-pass loss tolerance validation" begin
    loss(x) = (x[1] - 3.0)^2
    scan(x) = x[1]
    for method in (:CICO_ONE_PASS, :DIRECT_SLSQP), tolerance in (-1.0, Inf, NaN)
        @test_throws ArgumentError get_endpoint([3.0], 1, loss, method;
            loss_crit=4.0, loss_grad=:AUTODIFF, loss_tol=tolerance, silent=true)
        @test_throws ArgumentError get_endpoint([3.0], scan, loss, method;
            loss_crit=4.0, scan_grad=:AUTODIFF, loss_grad=:AUTODIFF,
            loss_tol=tolerance, silent=true)
    end
end

@testset "DIRECT_SLSQP with finite differences" begin
    test_alg_interval((algorithm=:LD_SLSQP, skip=[]);
        method=:DIRECT_SLSQP, bounds=(-1e10, 1e10),
        loss_grad=:FINITE, loss_tol=1e-8)
end
