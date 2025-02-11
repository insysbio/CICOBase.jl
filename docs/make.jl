
using CICOCore
using Documenter

makedocs(
    modules = [CICOCore],
    sitename = "CICOCore Documentation",
    pages = [
        "Home" => "index.md",
        "Methods" => "methods.md",
        "Visualization" => "visualization.md",
        "Optimizers" => "optimizers.md",
        "API" => "api.md",
    ],
)


deploydocs(
    repo   = "github.com/insysbio/CICOCore.jl.git",
    devbranch="master",
    devurl = "latest",
)
