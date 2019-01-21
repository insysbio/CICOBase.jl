
using Documenter, LikelihoodProfiler

makedocs(
    format = :html,
    #build   = "",
    sitename = "LikelihoodProfiler.jl",
    pages = [
        "index.md",
        "basics.md"
    ],
    #repo = "https://github.com/insysbio/LikelihoodProfiler.jl/blob/master/docs/src/index.md"
)

deploydocs(
    repo   = "github.com/insysbio/LikelihoodProfiler.jl.git",
    deps   = nothing,
    make = nothing,
    julia = "0.7",
    osname = "linux",
    target = "build"
)
