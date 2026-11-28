# Use the same cases, bounds and tolerances as the gradient-based series.
@testset "DIRECT_IPOPT with automatic gradients" begin
    test_alg_interval((algorithm=:DIRECT_IPOPT, skip=[]);
        method=:DIRECT_IPOPT, bounds=(-1e10, 1e10),
        loss_grad=:AUTODIFF, loss_tol=1e-8)
end

@testset "DIRECT_IPOPT with finite differences" begin
    test_alg_interval((algorithm=:DIRECT_IPOPT, skip=[]);
        method=:DIRECT_IPOPT, bounds=(-1e10, 1e10),
        loss_grad=:FINITE, loss_tol=1e-8)
end
