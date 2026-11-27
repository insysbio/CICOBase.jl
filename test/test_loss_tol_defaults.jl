@testset "method-specific loss tolerances" begin
    loss_defaults(x) = (x[1] - 3.0)^2
    scan_defaults(x) = x[1]
    for method in (:CICO_ONE_PASS, :DIRECT_SLSQP, :LIN_EXTRAPOL, :QUADR_EXTRAPOL)
        expected_tol = method in (:CICO_ONE_PASS, :DIRECT_SLSQP) ? 1e-3 : 0.0
        options = method == :DIRECT_SLSQP ? (; loss_grad=:AUTODIFF) : (;)
        implicit = get_interval([3.0], 1, loss_defaults, method;
            loss_crit=4.0, silent=true, options...)
        explicit = get_interval([3.0], 1, loss_defaults, method;
            loss_crit=4.0, loss_tol=expected_tol, silent=true, options...)
        @test [(p.status, p.value) for p in implicit.result] ==
              [(p.status, p.value) for p in explicit.result]
    end
    for method in (:CICO_ONE_PASS, :DIRECT_SLSQP)
        options = method == :DIRECT_SLSQP ? (; loss_grad=:AUTODIFF, scan_grad=:AUTODIFF) : (;)
        for tolerance in (-1.0, Inf, NaN)
            @test_throws ArgumentError get_endpoint([3.0], scan_defaults, loss_defaults, method;
                loss_crit=4.0, loss_tol=tolerance, silent=true, options...)
        end
        endpoint = get_endpoint([3.0], scan_defaults, loss_defaults, method;
            loss_crit=4.0, loss_tol=0.0, silent=true, options...)
        @test endpoint isa CICOBase.EndPoint
    end
end
