@testset "stable logit inversion" begin
    @test unscaling(-Inf, :logit) == 0.0
    @test unscaling(Inf, :logit) == 1.0
    @test unscaling(-400.0, :logit) == 0.0
    @test unscaling(400.0, :logit) == 1.0
    @test unscaling((-Inf, Inf), :logit) == (0.0, 1.0)

    for theta in (0.0, 0.01, 0.2, 0.5, 0.9, 0.99, 1.0)
        @test isapprox(unscaling(scaling(theta, :logit), :logit), theta)
    end

    for x in (-400.0, -2.0, 0.0, 2.0, 400.0)
        theta = unscaling(x, :logit)
        derivative = CICOBase.ForwardDiff.derivative(t -> unscaling(t, :logit), x)
        @test isfinite(derivative)
        @test isapprox(derivative, log(10.0) * theta * (1 - theta))
    end

    # Default logit bounds must be finite values in the original scale.
    pp = get_optimal([0.25], p -> (p[1] - 0.6)^2;
        scale = [:logit], loss_tol = 1e-10, max_iter = 1000, silent = true)
    @test isapprox(pp.params[1], 0.6; atol = 1e-3)
    @test pp.loss < 1e-6
end
