function err_fun_generate()
    counter::Integer = 0
    fun = (x::Vector{Float64}) -> begin
        counter += 1
        if counter > 5
            throw(ErrorException("Function can be called only 5 times."))
        end
        5.0 + (x[1]-3.0)^2 + (x[2]-4.0)^2
    end
end

@testset "callback error sources in :CICO_ONE_PASS" begin
    loss(x) = -1.0
    scan(x) = x[1]
    good_grad(x) = [1.0]
    bad_grad(x) = error("gradient failed")
    bad_scan(x) = error("scan failed")

    @test get_right_endpoint([0.0], bad_scan, loss, Val(:CICO_ONE_PASS);
        scan_bound = 9.0)[3] == :SCAN_ERROR_STOP
    @test get_right_endpoint([0.0], scan, loss, Val(:CICO_ONE_PASS);
        local_alg = :LD_MMA, scan_grad = good_grad, loss_grad = bad_grad)[3] == :LOSS_GRAD_ERROR_STOP
    @test get_right_endpoint([0.0], scan, loss, Val(:CICO_ONE_PASS);
        local_alg = :LD_MMA, scan_grad = bad_grad, loss_grad = good_grad)[3] == :SCAN_GRAD_ERROR_STOP
    forced_scan(x) = throw(NLopt.ForcedStop())
    @test get_right_endpoint([0.0], forced_scan, loss, Val(:CICO_ONE_PASS))[3] == :NLOPT_FORCED_STOP
end

@testset "loss error in :CICO_ONE_PASS" begin
    err_fun = err_fun_generate()
    res0 = get_endpoint(
        [3., 4.],
        1,
        err_fun,
        :CICO_ONE_PASS;
        loss_crit = 9.,
        silent = true
    )
    @test res0.value === nothing
    @test res0.profilePoints == ProfilePoint[]
    @test res0.status == :LOSS_ERROR_STOP
    @test res0.counter < 5
    @test typeof(res0.supreme) == Float64
end

@testset "test profile error" begin
    err_func = err_fun_generate()
    prof = CICOBase._profile(
        [3., 4.],
        1,
        err_func
    )
    res0 = prof(5.)
    @test res0.ret == :FORCED_STOP
    @test res0.force_stop_code == CICOBase.STOP_CODE_LOSS_ERROR
    @test res0.counter == 5
end

@testset "profile can run again after forced stop" begin
    fail = Ref(true)
    loss(x) = fail[] ? error("temporary loss error") : (x[2] - 1.0)^2
    prof = CICOBase._profile([0.0, 0.0], 1, loss)
    first = prof(0.0)
    @test first.ret == :FORCED_STOP
    @test first.force_stop_code == CICOBase.STOP_CODE_LOSS_ERROR
    fail[] = false
    second = prof(0.0)
    @test second.ret != :FORCED_STOP
    @test second.force_stop_code == 0
end

@testset "unclassified forced stop in extrapolation" begin
    forced_loss(x) = throw(NLopt.ForcedStop())
    for method in (:LIN_EXTRAPOL, :QUADR_EXTRAPOL)
        _, points, status = get_right_endpoint([0.0, 0.0], 1, forced_loss, Val(method))
        @test status == :NLOPT_FORCED_STOP
        @test points[1].ret == :FORCED_STOP
        @test points[1].force_stop_code == 1
    end
end

@testset "loss error in :LIN_EXTRAPOL" begin
    err_fun = err_fun_generate()
    res0 = get_endpoint(
        [3., 4.],
        1,
        err_fun,
        :LIN_EXTRAPOL;
        loss_crit = 9.,
        silent = true
    )
    @test res0.value === nothing
    @test length(res0.profilePoints) == 1
    @test res0.status == :LOSS_ERROR_STOP
    @test res0.counter < 5
    @test typeof(res0.supreme) == Float64
    pp = res0.profilePoints[1]
    @test pp.ret == :FORCED_STOP
    @test pp.force_stop_code == CICOBase.STOP_CODE_LOSS_ERROR
end

@testset "loss error in :QUADR_EXTRAPOL" begin
    err_fun = err_fun_generate()
    res0 = get_endpoint(
        [3., 4.],
        1,
        err_fun,
        :QUADR_EXTRAPOL;
        loss_crit = 9.,
        silent = true
    )
    @test res0.value === nothing
    @test length(res0.profilePoints) == 1
    @test res0.status == :LOSS_ERROR_STOP
    @test res0.counter < 5
    @test typeof(res0.supreme) == Float64
    pp = res0.profilePoints[1]
    @test pp.ret == :FORCED_STOP
    @test pp.force_stop_code == CICOBase.STOP_CODE_LOSS_ERROR
end
