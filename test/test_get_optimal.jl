# tests for `get_optimal` method
# testing only the case f_2p
# using CICOBase

@testset "automatic and explicit initial step" begin
    loss(x) = (x[1] - 2.0)^2 + (x[2] - 3.0)^2
    initial = [0.0, 1.0]

    result = get_optimal(initial, loss; scan_tol = 1e-5, silent = true)
    @test isapprox(result.params, [2.0, 3.0]; atol = 1e-3)

    @test isapprox(get_optimal(initial, loss; local_initial_step = 1.0,
                               scan_tol = 1e-5, silent = true).loss, 0.0; atol = 1e-5)
    @test isapprox(get_optimal(initial, loss; local_initial_step = [1.0, 2.0],
                               scan_tol = 1e-5, silent = true).loss, 0.0; atol = 1e-5)
    @test_throws ArgumentError get_optimal(initial, loss; local_initial_step = 0.0)
    @test_throws ArgumentError get_optimal(initial, loss; local_initial_step = [1.0, 0.0])
end

@testset "testing get_optimal() for :LN_PRAXIS and f_2p, identifiable" begin
    res0 = get_optimal(
        [4., 1.],
        f_2p;
        silent = true,
        loss_tol = 1e-3,
        scan_tol = 0.,
        local_alg = :LN_PRAXIS
    )
    @test isapprox(res0.loss, 5., atol=1e-3)
    @test res0.ret == :FTOL_REACHED

    res1 = get_optimal(
        [4., 1.],
        f_2p;
        silent = true,
        loss_tol = 0.,
        scan_tol = 1e-3,
        local_alg = :LN_PRAXIS
    )
    @test isapprox(res1.params[1], 3., atol=1e-3)
    @test isapprox(res1.params[2], 4., atol=1e-3)
    @test res1.ret == :XTOL_REACHED

    res2 = get_optimal(
        [4., 1.],
        f_2p;
        silent = true,
        loss_tol = 0.,
        scan_tol = 0.,
        max_iter = 10,
        local_alg = :LN_PRAXIS
    )
    @test res2.ret == :MAXEVAL_REACHED

    res3 = get_optimal(
        [4., 1.],
        f_2p;
        silent = true,
        loss_tol = 1e-3,
        scan_tol = 0.,
        local_alg = :LN_PRAXIS,
        scale = [:log, :log]
    )
    @test isapprox(res3.loss, 5., atol=1e-3)
    @test res3.ret == :FTOL_REACHED
end

@testset "testing get_optimal() for :LN_PRAXIS and f_2p, nonidentifiable" begin
    res0 = get_optimal(
        [4., 1., 1.],
        f_2p;
        silent = true,
        loss_tol = 1e-3,
        scan_tol = 0.,
        local_alg = :LN_PRAXIS
    )
    @test isapprox(res0.loss, 5., atol=1e-3)
    @test res0.ret == :FTOL_REACHED

    res1 = get_optimal(
        [4., 1., 1.],
        f_2p;
        silent = true,
        loss_tol = 0.,
        scan_tol = 1e-3,
        local_alg = :LN_PRAXIS
    )
    @test isapprox(res1.params[1], 3., atol=1e-3)
    @test isapprox(res1.params[2], 4., atol=1e-3)
    @test res1.ret == :XTOL_REACHED

    res2 = get_optimal(
        [4., 1., 1.],
        f_2p;
        silent = true,
        loss_tol = 0.,
        scan_tol = 0.,
        max_iter = 10,
        local_alg = :LN_PRAXIS
    )
    @test res2.ret == :MAXEVAL_REACHED

    res3 = get_optimal(
        [4., 1., 1.],
        f_2p;
        silent = true,
        loss_tol = 1e-3,
        scan_tol = 0.,
        local_alg = :LN_PRAXIS,
        scale = [:log, :log, :log]
    )
    @test isapprox(res3.loss, 5., atol=1e-3)
    @test res3.ret == :FTOL_REACHED
end

#f_2p(x) = 5.0 + (x[1]-3.0)^2 + (x[2]-4.0)^2
grad_2p(x) = [2*(x[1]-3.), 2*(x[2]-4.), 0.]

@testset "testing get_optimal() for :LD_MMA and f_2p, nonidentifiable" begin
    res0 = get_optimal(
        [4., 1., 1.],
        f_2p;
        silent = true,
        loss_tol = 1e-3,
        scan_tol = 0.,
        local_alg = :LD_MMA,
        loss_grad = grad_2p
    )
    @test isapprox(res0.loss, 5., atol=1e-3)
    @test res0.ret == :FTOL_REACHED

    res1 = get_optimal(
        [4., 1., 1.],
        f_2p;
        silent = true,
        loss_tol = 0.,
        scan_tol = 1e-3,
        local_alg = :LD_MMA,
        loss_grad = grad_2p
    )
    @test isapprox(res1.params[1], 3., atol=1e-3)
    @test isapprox(res1.params[2], 4., atol=1e-3)
    @test res1.ret == :XTOL_REACHED

    res2 = get_optimal(
        [4., 1., 1.],
        f_2p;
        silent = true,
        loss_tol = 0.,
        scan_tol = 0.,
        max_iter = 10,
        local_alg = :LD_MMA,
        loss_grad = grad_2p
    )
    @test res2.ret == :MAXEVAL_REACHED

    res3 = get_optimal(
        [4., 1., 1.],
        f_2p;
        silent = true,
        loss_tol = 1e-3,
        scan_tol = 0.,
        local_alg = :LD_MMA,
        scale = [:log, :log, :log],
        loss_grad = grad_2p
    )
    @test isapprox(res3.loss, 5., atol=1e-3)
    @test res3.ret == :FTOL_REACHED
end

@testset "testing get_optimal() for :LD_MMA and f_2p, nonidentifiable, finite" begin
    res0 = get_optimal(
        [4., 1., 1.],
        f_2p;
        silent = true,
        loss_tol = 1e-3,
        scan_tol = 0.,
        local_alg = :LD_MMA,
        loss_grad = :FINITE # :AUTODIFF
    )
    @test isapprox(res0.loss, 5., atol=1e-3)
    @test res0.ret == :FTOL_REACHED

    res1 = get_optimal(
        [4., 1., 1.],
        f_2p;
        silent = true,
        loss_tol = 0.,
        scan_tol = 1e-3,
        local_alg = :LD_MMA,
        loss_grad = :FINITE
    )
    @test isapprox(res1.params[1], 3., atol=1e-3)
    @test isapprox(res1.params[2], 4., atol=1e-3)
    @test res1.ret == :XTOL_REACHED

    res2 = get_optimal(
        [4., 1., 1.],
        f_2p;
        silent = true,
        loss_tol = 0.,
        scan_tol = 0.,
        max_iter = 10,
        local_alg = :LD_MMA,
        loss_grad = :FINITE
    )
    @test res2.ret == :MAXEVAL_REACHED

    res3 = get_optimal(
        [4., 1., 1.],
        f_2p;
        silent = true,
        loss_tol = 1e-3,
        scan_tol = 0.,
        local_alg = :LD_MMA,
        scale = [:log, :log, :log],
        loss_grad = :FINITE
    )
    @test isapprox(res3.loss, 5., atol=1e-3)
    @test res3.ret == :FTOL_REACHED
end

@testset "testing get_optimal() for :LD_MMA and f_2p, nonidentifiable, autodiff" begin
    res0 = get_optimal(
        [4., 1., 1.],
        f_2p;
        silent = true,
        loss_tol = 1e-3,
        scan_tol = 0.,
        local_alg = :LD_MMA,
        loss_grad = :AUTODIFF
    )
    @test isapprox(res0.loss, 5., atol=1e-3)
    @test res0.ret == :FTOL_REACHED

    res1 = get_optimal(
        [4., 1., 1.],
        f_2p;
        silent = true,
        loss_tol = 0.,
        scan_tol = 1e-3,
        local_alg = :LD_MMA,
        loss_grad = :AUTODIFF
    )
    @test isapprox(res1.params[1], 3., atol=1e-3)
    @test isapprox(res1.params[2], 4., atol=1e-3)
    @test res1.ret == :XTOL_REACHED

    res2 = get_optimal(
        [4., 1., 1.],
        f_2p;
        silent = true,
        loss_tol = 0.,
        scan_tol = 0.,
        max_iter = 10,
        local_alg = :LD_MMA,
        loss_grad = :AUTODIFF
    )
    @test res2.ret == :MAXEVAL_REACHED

    res3 = get_optimal(
        [4., 1., 1.],
        f_2p;
        silent = true,
        loss_tol = 1e-3,
        scan_tol = 0.,
        local_alg = :LD_MMA,
        scale = [:log, :log, :log],
        loss_grad = :AUTODIFF
    )
    @test isapprox(res3.loss, 5., atol=1e-3)
    @test res3.ret == :FTOL_REACHED
end

@testset "FINITE handles objective-only requests" begin
    target = [3.0, 0.4]
    loss(p) = sum(abs2, p .- target)
    for scale in ([:direct, :direct], [:log, :logit]),
        local_alg in (:LD_SLSQP, :LN_NELDERMEAD)
        pp = get_optimal([4.0, 0.25], loss;
            scale, local_alg, loss_grad = :FINITE,
            loss_tol = 1e-10, scan_tol = 1e-8,
            max_iter = 1000, silent = true
        )
        @test isapprox(pp.params, target; atol = 1e-3)
        @test pp.loss < 1e-6
        @test pp.loss == loss(pp.params)
    end
end
