loss_penalty_test(x) = (x[1] - 3.0)^2
scan_penalty_test(x) = x[1]

for scan_tol in (1e-3, 1e-5)
    implicit = get_interval([3.0], 1, loss_penalty_test, :CICO_ONE_PASS;
        loss_crit=4.0, scan_tol, silent=true)
    explicit = get_interval([3.0], 1, loss_penalty_test, :CICO_ONE_PASS;
        loss_crit=4.0, scan_tol, penalty_tol=1e-6, silent=true)
    @test [(p.status, p.value) for p in implicit.result] ==
          [(p.status, p.value) for p in explicit.result]
end

for tolerance in (-1.0, Inf, NaN)
    @test_throws ArgumentError get_endpoint([3.0], 1, loss_penalty_test, :CICO_ONE_PASS;
        loss_crit=4.0, penalty_tol=tolerance, silent=true)
    @test_throws ArgumentError get_endpoint([3.0], scan_penalty_test, loss_penalty_test, :CICO_ONE_PASS;
        loss_crit=4.0, penalty_tol=tolerance, silent=true)
end
