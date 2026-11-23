# Standalone experiment: julia --project test/experiment_interval_validation.jl
# Not included by runtests.jl. Output goes to draft/interval-validation-experiment.
using CICOBase, Test, NLopt
include("cases_func.jl")

const EXP_ALGORITHMS = Symbol(get(ENV, "CICO_EXPERIMENT_ALGORITHMS", "derivative_free"))
EXP_ALGORITHMS in (:derivative_free, :gradient) || error("CICO_EXPERIMENT_ALGORITHMS must be derivative_free or gradient")
const ALGORITHMS = EXP_ALGORITHMS == :gradient ?
    [:LD_MMA, :LD_SLSQP, :LD_CCSAQ, :LD_LBFGS, :LD_TNEWTON_PRECOND_RESTART,
     :LD_TNEWTON_PRECOND, :LD_TNEWTON_RESTART, :LD_TNEWTON, :LD_VAR2, :LD_VAR1] :
    [:LN_NELDERMEAD, :LN_PRAXIS, :LN_SBPLX, :LN_NEWUOA, :LN_BOBYQA, :LN_COBYLA]
const ROOT = dirname(@__DIR__)
const PROJECT_ROOT = get(ENV, "CICO_EXPERIMENT_PROJECT", ROOT)
const OUT = get(ENV, "CICO_EXPERIMENT_OUT", joinpath(ROOT, "draft", "interval-validation-experiment"))
const BOUNDARY_ATOL = 1e-2
const PROFILE_ATOL = 1e-2
const REFERENCE_ATOL = 10 * TOL # matches test_alg_interval's endpoint check
const MAX_ITER = 10^6
const INITIAL_STEP_TEXT = get(ENV, "CICO_EXPERIMENT_INITIAL_STEP", "nothing")
const INITIAL_STEP = INITIAL_STEP_TEXT == "nothing" ? nothing : parse(Float64, INITIAL_STEP_TEXT)
const TIMEOUT = 60.0 # wall seconds per interval and its validation, after worker warmup
const EXP_SCALE = Symbol(get(ENV, "CICO_EXPERIMENT_SCALE", "direct"))
EXP_SCALE in (:direct, :log) || error("CICO_EXPERIMENT_SCALE must be direct or log")
const THETA_LIMIT = parse(Float64, get(ENV, "CICO_EXPERIMENT_THETA_LIMIT", "Inf"))
THETA_LIMIT > 9 && !isnan(THETA_LIMIT) || error("CICO_EXPERIMENT_THETA_LIMIT must be greater than 9")
EXP_SCALE == :direct || isinf(THETA_LIMIT) || error("A finite theta limit is supported only with direct scale")
const WORKERS = 3
const RESULT_IO = Ref{IO}(stdout)

# Reference values are used only for the subsequent independent comparison;
# they are never passed to the computation or the validator.
cases = Dict{Symbol,NamedTuple}()
for (name, f) in test_funcs
    cases[name] = (func=f.func, start=f.x1, crit=f.loss_crit, source="cases_func.jl")
end
cases[:noderiv_f1] = (func=x -> 3x[1]^4 - 6x[1]^3 + 5, start=[1.5], crit=5.0, source="test_noderiv.jl")
cases[:noderiv_f2] = (func=x -> -3(x[1]-5)^4 + 5(x[1]-5)^2 + 5, start=[5.0], crit=7.0, source="test_noderiv.jl")
cases[:noderiv_f3] = (func=x -> 3x[1]^3 - 6x[1]^2 + 5, start=[1.5], crit=5.0, source="test_noderiv.jl")
cases[:noderiv_f4] = (func=x -> 3x[1]^4 - 6x[1]^3 - x[1]^2 + x[1] + 5, start=[1.5], crit=5.0, source="test_noderiv.jl")
# f5 is defined but not run in test_noderiv.jl; its starting point and level
# follow the adjacent f1/f3/f4 examples and satisfy the initial-loss condition.
cases[:noderiv_f5] = (func=x -> 3x[1]^4 - 6x[1]^3 - 3x[1]^2 + 4x[1] + 5, start=[1.5], crit=5.0, source="test_noderiv.jl")
cases[:noderiv_f6] = (func=x -> 1.0 <= x[1] < 5.0 ? (x[1]-3)^2 + 5.0 : abs(4.5x[1]-13.5), start=[3.0], crit=9.0, source="test_noderiv.jl")
cases[:scan_example_loss] = (func=x -> (x[1]-5)^2 + (x[2]-6)^2, start=[2.0,2.0,2.0], crit=64.0, source="test_get_endpoint_scan_func.jl")
cases[:interval_scan_loss] = (func=x -> (x[1]-2)^2 + (x[2]-0.5)^2, start=[2.0,0.5], crit=0.01, source="test_get_interval.jl")
cases[:scan_bound_loss] = (func=x -> (x[1]-2)^2 + 0.0*x[2], start=[2.0,2.0], crit=1.0, source="test_cico_one_pass.jl")

function clean(x)
    replace(string(x), '\t'=>' ', '\n'=>' ', '\r'=>' ')
end
function retry_read(path)
    for attempt in 1:100
        try
            return read(path, String)
        catch e
            (e isa SystemError || e isa Base.IOError) || rethrow()
            attempt == 100 && rethrow()
            sleep(0.05)
        end
    end
end
function emit(alg, name, index, side, status, outcome, reason, value, loss, residual, improvement, detail)
    println(RESULT_IO[], "RESULT\t", join(clean.((alg, name, index, side, status, outcome, reason,
                                      value, loss, residual, improvement, detail)), '\t'))
    flush(RESULT_IO[])
end

function worker(alg, name, index)
    f = cases[name]
    NLopt.srand(12345)
    try
        interval = get_interval(f.start, index, f.func, :CICO_ONE_PASS;
            theta_bounds=fill(EXP_SCALE == :log ? (0.0, Inf) : (-THETA_LIMIT, THETA_LIMIT), length(f.start)),
            scale=fill(EXP_SCALE, length(f.start)), scan_tol=1e-8,
            loss_crit=f.crit, local_alg=alg, max_iter=MAX_ITER,
            local_initial_step=INITIAL_STEP,
            loss_grad=EXP_ALGORITHMS == :gradient ? :AUTODIFF : :EMPTY, silent=true)
        result = validate_interval(interval; boundary_loss_atol=BOUNDARY_ATOL,
                                   profile_loss_atol=PROFILE_ATOL)
        for r in (result.left, result.right)
            improvement = isnothing(r.profile) ? nothing : r.profile.improvement
            emit(alg, name, index, r.direction, r.original_status, r.outcome, r.reason,
                 r.endpoint, r.loss, r.boundary_residual, improvement, "")
        end
    catch e
        e isa InterruptException && rethrow()
        for side in (:left, :right)
            emit(alg, name, index, side, "EXCEPTION", "error", "exception",
                 nothing, nothing, nothing, nothing, sprint(showerror, e))
        end
    end
end

if !isempty(ARGS) && ARGS[1] == "--server"
    slot = parse(Int, ARGS[2])
    folder = ARGS[3]
    request = joinpath(folder, "slot$(slot).request")
    response = joinpath(folder, "slot$(slot).response")
    ready = joinpath(folder, "slot$(slot).ready")
    # Compile the common paths before reporting readiness; this time is not
    # charged to the first interval. Output is deliberately discarded.
    RESULT_IO[] = devnull
    worker(first(ALGORITHMS), :f_2p, 1)
    write(ready, "ready")
    while true
        if isfile(request)
            fields = split(retry_read(request), '\t')
            rm(request)
            fields[1] == "STOP" && break
            temporary = response * ".tmp"
            open(temporary, "w") do io
                RESULT_IO[] = io
                worker(Symbol(fields[1]), Symbol(fields[2]), parse(Int, fields[3]))
            end
            mv(temporary, response; force=true)
        end
        sleep(0.05)
    end
    exit()
end

mkpath(OUT)
run_dir = mktempdir(OUT; prefix="run-", cleanup=false)
jobs = [(alg, name, i) for alg in ALGORITHMS for name in sort!(collect(keys(cases)))
        for i in eachindex(cases[name].start)]
pending = copy(jobs)
results = Dict{Tuple{Symbol,Symbol,Int}, Vector{String}}()
started = time()
function start_server(slot)
    io = open(joinpath(run_dir, "slot$(slot).log"), "a")
    cmd = `$(Base.julia_cmd()) --project=$PROJECT_ROOT $(@__FILE__) --server $slot $run_dir`
    proc = run(pipeline(cmd, stdout=io, stderr=io); wait=false)
    return (process=proc, io=io, job=nothing, start=time())
end
slots = Any[start_server(i) for i in 1:WORKERS]
while length(results) < length(jobs)
    sleep(0.2)
    for slot in 1:WORKERS
        task = slots[slot]
        ready = joinpath(run_dir, "slot$(slot).ready")
        request = joinpath(run_dir, "slot$(slot).request")
        response = joinpath(run_dir, "slot$(slot).response")
        if !isnothing(task.job) && isfile(response)
            rows = filter(line -> startswith(line, "RESULT\t"), split(chomp(retry_read(response)), '\n'))
            rm(response)
            if length(rows) != 2
                error("Worker $slot returned $(length(rows)) result rows for $(task.job)")
            end
            results[task.job] = rows
            println("$(length(results))/$(length(jobs)) $(task.job) done")
            flush(stdout)
            slots[slot] = (process=task.process, io=task.io, job=nothing, start=time())
        elseif !process_running(task.process) || (!isnothing(task.job) && time()-task.start > TIMEOUT)
            timed_out = process_running(task.process)
            timed_out && kill(task.process)
            wait(task.process)
            close(task.io)
            if !isnothing(task.job)
                alg, name, index = task.job
                category = timed_out ? "timeout" : "error"
                reason = timed_out ? "wall_timeout" : "worker_exit_$(task.process.exitcode)"
                results[task.job] = [join(("RESULT", alg, name, index, side, "NO_RESULT", category,
                    reason, "nothing", "nothing", "nothing", "nothing", "slot$(slot).log"), '\t')
                    for side in (:left, :right)]
                println("$(length(results))/$(length(jobs)) $(task.job) $category")
                flush(stdout)
            else
                error("Worker $slot failed during startup; see $(joinpath(run_dir, "slot$(slot).log"))")
            end
            for path in (ready, request, response, response * ".tmp")
                isfile(path) && rm(path)
            end
            slots[slot] = start_server(slot)
        elseif isnothing(task.job) && isfile(ready) && !isempty(pending)
            job = popfirst!(pending)
            temporary = request * ".tmp"
            write(temporary, join(job, '\t'))
            mv(temporary, request; force=true)
            slots[slot] = (process=task.process, io=task.io, job=job, start=time())
        end
    end
end
for slot in 1:WORKERS
    task = slots[slot]
    process_running(task.process) && kill(task.process)
    wait(task.process)
    close(task.io)
end

csvfield(value) = '"' * replace(string(value), '"' => "\"\"") * '"'
csvrow(io, values) = println(io, join(csvfield.(values), ','))
function compare_reference(job, line)
    _, name, index = job
    row = split(line, '\t'; keepempty=true)
    if !haskey(test_funcs, name)
        return (status="", value="", verdict="no_reference", reason="no_test_oracle", error="")
    end
    side = row[5] == "left" ? 1 : 2
    expected_status = test_funcs[name].status[index][side]
    expected_value = test_funcs[name].endpoints[index][side]
    if EXP_SCALE == :log && side == 1 &&
       ((name == :f_1p_ex && index == 1) || (name == :f_3p_1im_dep && index == 2))
        # The direct-scale lower root is negative. In these two analytic
        # examples loss remains below the critical level at 1e-9 with all
        # parameters positive, so the log-scale lower scan bound is reached.
        expected_status = :SCAN_BOUND_REACHED
        expected_value = nothing
    end
    actual_status = Symbol(row[6])
    if row[7] in ("timeout", "error")
        return (status=expected_status, value=expected_value, verdict="unavailable", reason=row[7], error="")
    end
    actual_value = row[9] == "nothing" ? nothing : tryparse(Float64, row[9])
    error_value = isnothing(expected_value) || isnothing(actual_value) ? "" : abs(actual_value - expected_value)
    status_ok = actual_status == expected_status
    value_ok = isnothing(expected_value) ? isnothing(actual_value) :
        !isnothing(actual_value) && isfinite(actual_value) && error_value <= REFERENCE_ATOL
    verdict = status_ok && value_ok ? "correct" : "incorrect"
    reason = !status_ok ? "status_mismatch" : !value_ok ? "value_mismatch" : "matches"
    return (status=expected_status, value=expected_value, verdict, reason, error=error_value)
end
open(joinpath(run_dir, "results.csv"), "w") do io
    print(io, '\ufeff') # UTF-8 BOM for Excel on Windows
    csvrow(io, ("algorithm", "function", "parameter", "side", "status", "outcome",
        "reason", "endpoint", "loss", "boundary_residual", "profile_improvement", "detail",
        "reference_status", "reference_endpoint", "reference_verdict", "reference_reason", "reference_abs_error"))
    for job in jobs, line in results[job]
        ref = compare_reference(job, line)
        csvrow(io, (split(line, '\t'; keepempty=true)[2:end]..., ref.status, ref.value, ref.verdict, ref.reason, ref.error))
    end
end
open(joinpath(run_dir, "reference_comparison.csv"), "w") do io
    print(io, '\ufeff')
    csvrow(io, ("algorithm", "validator_pass_reference_correct", "validator_pass_reference_incorrect",
        "validator_fail_reference_correct", "validator_fail_reference_incorrect",
        "validator_inconclusive_reference_correct", "validator_inconclusive_reference_incorrect",
        "validator_not_checked_reference_correct", "validator_not_checked_reference_incorrect",
        "reference_unavailable", "no_reference"))
    for alg in ALGORITHMS
        records = [(split(line, '\t'; keepempty=true)[7], compare_reference(job, line).verdict)
            for job in jobs if job[1] == alg for line in results[job]]
        counts = [count(==((v, ref)), records) for (v, ref) in
            (("pass", "correct"), ("pass", "incorrect"), ("fail", "correct"), ("fail", "incorrect"),
             ("inconclusive", "correct"), ("inconclusive", "incorrect"),
             ("not_checked", "correct"), ("not_checked", "incorrect"))]
        unavailable = count(r -> r[2] == "unavailable", records)
        no_reference = count(r -> r[2] == "no_reference", records)
        csvrow(io, (alg, counts..., unavailable, no_reference))
    end
end
open(joinpath(run_dir, "reference_mismatches.csv"), "w") do io
    print(io, '\ufeff')
    csvrow(io, ("algorithm", "function", "parameter", "side", "status", "outcome",
        "reason", "endpoint", "loss", "boundary_residual", "profile_improvement", "detail",
        "reference_status", "reference_endpoint", "reference_verdict", "reference_reason", "reference_abs_error"))
    for job in jobs, line in results[job]
        ref = compare_reference(job, line)
        ref.verdict == "incorrect" || continue
        row = split(line, '\t'; keepempty=true)
        csvrow(io, (row[2:end]..., ref.status, ref.value, ref.verdict, ref.reason, ref.error))
    end
end
open(joinpath(run_dir, "summary.csv"), "w") do io
    print(io, '\ufeff')
    csvrow(io, ("Алгоритм", "Проверка пройдена", "Не пройдена", "Неопределённо", "Не проверено", "Превышено время", "Ошибка", "Всего сторон"))
    for alg in ALGORITHMS
        rows = [split(line, '\t'; keepempty=true) for job in jobs if job[1] == alg for line in results[job]]
        counts = [count(row -> row[7] == outcome, rows) for outcome in ("pass", "fail", "inconclusive", "not_checked", "timeout", "error")]
        csvrow(io, (alg, counts..., length(rows)))
    end
end
open(joinpath(run_dir, "details.csv"), "w") do io
    print(io, '\ufeff')
    headings = ["Функция", "Параметр"]
    for alg in ALGORITHMS
        append!(headings, ["$alg левая", "$alg правая"])
    end
    csvrow(io, headings)
    labels = Dict("pass"=>"пройдена", "fail"=>"не пройдена", "inconclusive"=>"неопределённо", "not_checked"=>"не проверено", "timeout"=>"превышено время", "error"=>"ошибка")
    for name in sort!(collect(keys(cases))), index in eachindex(cases[name].start)
        cells = String[]
        for alg in ALGORITHMS
            records = [split(line, '\t'; keepempty=true) for line in results[(alg, name, index)]]
            append!(cells, [get(labels, row[7], "unknown") for row in records])
        end
        csvrow(io, (name, index, cells...))
    end
end
open(joinpath(run_dir, "cases.csv"), "w") do io
    print(io, '\ufeff')
    csvrow(io, ("function", "source", "initial_parameters", "loss_crit", "parameter_count"))
    for name in sort!(collect(keys(cases)))
        f = cases[name]
        csvrow(io, (name, f.source, f.start, f.crit, length(f.start)))
    end
end
open(joinpath(run_dir, "protocol.md"), "w") do io
    println(io, "# Interval validation experiment\n")
    println(io, "local_initial_step=$INITIAL_STEP (NLopt may ignore it for some local algorithms).\n")
    println(io, "CICO_ONE_PASS; algorithms=$EXP_ALGORITHMS; loss_grad=$(EXP_ALGORITHMS == :gradient ? ":AUTODIFF (ForwardDiff)" : ":EMPTY"); scan gradient for parameter scans is analytic; validation profile optimizer is the default derivative-free LN_NELDERMEAD. Parameter scale=$EXP_SCALE; scan_tol=1e-8 in optimization coordinates; scan bounds $(EXP_SCALE == :log ? "(1e-9, 1e9)" : "(-9, 9)"); theta bounds $(EXP_SCALE == :log ? "(0, Inf)" : "(-$THETA_LIMIT, $THETA_LIMIT)"); max_iter=$MAX_ITER; loss tolerances boundary/profile = $BOUNDARY_ATOL/$PROFILE_ATOL. Validator defaults otherwise. Seed 12345 per case. $TIMEOUT seconds per interval and validation after Julia worker warmup. One-sided outcomes are counted independently; timeouts count both sides.\n")
    println(io, "Reference comparison uses analytic statuses and endpoints from cases_func.jl with absolute endpoint tolerance $REFERENCE_ATOL in original parameter units; cases without explicit references are no_reference. For log scale, the lower negative roots of f_1p_ex parameter 1 and f_3p_1im_dep parameter 2 are replaced with SCAN_BOUND_REACHED at the positive scan bound; these are analytically inferred expectations, not the original direct-scale test assertions. Other scan bounds also change from +/-9 to 1e-9/1e9; the reference assumes the original analytic case properties extend to those bounds. Reference is not used by the validator.\n")
    println(io, "Pass is a local validator check, not proof of global correctness. Only BORDER_FOUND_BY_SCAN_TOL is validated; SCAN_BOUND_REACHED and BORDER_FOUND_BY_LOSS_TOL are not_checked. Prediction scans are not supported by the validator. Callback-error examples are excluded. noderiv_f5 was defined without a call; its start and threshold follow adjacent examples.\n")
    println(io, "Wall time: $(round(time()-started; digits=1)) seconds.")
end
println("Results: ", run_dir)
