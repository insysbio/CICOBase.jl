@testset "DIRECT_IPOPT derivative interface" begin
    loss(x) = (x[1] - 2)^2 + (x[2] - 3)^2
    grad(x) = [2(x[1] - 2), 2(x[2] - 3)]
    hessian_calls = Ref(0)
    function loss_hessian(x)
        hessian_calls[] += 1
        return [2.0 0.0; 0.0 2.0]
    end
    for options in ((; loss_grad=:AUTODIFF, hessian=:AUTODIFF),
                    (; loss_grad=grad, hessian=:LIMITED_MEMORY),
                    (; loss_grad=grad, hessian=loss_hessian),
                    (; loss_grad=:AUTODIFF, hessian=loss_hessian),
                    (; loss_grad=:AUTODIFF))
        interval = get_interval([2.0, 3.0], 1, loss, :DIRECT_IPOPT;
            loss_crit=0.25, scale=[:log, :log], theta_bounds=[(0.0, Inf), (0.0, Inf)],
            scan_bounds=(0.5, 5.0), scan_tol=1e-8, loss_tol=1e-8, silent=true, options...)
        @test all(p.status == :BORDER_FOUND_BY_SCAN_TOL for p in interval.result)
        @test interval.result[1].value ≈ 1.5 atol=1e-4
        @test interval.result[2].value ≈ 2.5 atol=1e-4
    end
    @test hessian_calls[] > 0
    scan(x) = x[1]^2 + x[2]
    sphere(x) = sum(abs2, x)
    endpoint = get_endpoint([0.2, 0.0], scan, sphere, :DIRECT_IPOPT;
        loss_crit=1.0, scan_grad=:AUTODIFF, loss_grad=:AUTODIFF,
        hessian=:AUTODIFF, scan_tol=1e-8, loss_tol=1e-8, silent=true)
    @test endpoint.status == :BORDER_FOUND_BY_SCAN_TOL
    @test endpoint.value ≈ 1.25 atol=1e-5
    supplied = get_endpoint([0.2, 0.0], scan, sphere, :DIRECT_IPOPT;
        loss_crit=1.0, scan_grad=:AUTODIFF, loss_grad=:AUTODIFF,
        hessian=loss_hessian, scan_tol=1e-8, loss_tol=1e-8, silent=true)
    @test supplied.status == :BORDER_FOUND_BY_SCAN_TOL
    @test supplied.value ≈ 1.25 atol=1e-5
    # Cross terms and second derivatives of both scales, including left-side sign.
    coupled(x) = (x[1] - 2)^2 + 0.6(x[1] - 2) * (x[2] - 0.4) + (x[2] - 0.4)^2
    coupled_grad(x) = [2(x[1] - 2) + 0.6(x[2] - 0.4), 0.6(x[1] - 2) + 2(x[2] - 0.4)]
    coupled_hessian(x) = [2.0 0.6; 0.6 2.0]
    z = [-log10(1.8), log10(0.3 / 0.7)]
    transformed(z) = coupled([unscaling(-z[1], :log), unscaling(z[2], :logit)])
    actual_hessian = CICOBase._transform_ipopt_loss_hessian(
        coupled_hessian, coupled, coupled_grad, [:log, :logit], 1)(z)
    @test actual_hessian ≈ CICOBase.ForwardDiff.hessian(transformed, z) atol=1e-9
    @test_throws ArgumentError get_endpoint([2.0, 3.0], 1, loss, :DIRECT_IPOPT;
        loss_crit=0.25, loss_grad=:AUTODIFF, hessian=:INVALID, silent=true)
end
