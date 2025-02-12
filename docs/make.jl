
using CICOBase
using Documenter

makedocs(
    modules = [CICOBase],
    sitename = "CICOBase Documentation",
    pages = [
        "Home" => "index.md",
        "Methods" => "methods.md",
        "Visualization" => "visualization.md",
        "Optimizers" => "optimizers.md",
        "API" => "api.md",
    ],
)


deploydocs(
    repo   = "github.com/insysbio/CICOBase.jl.git",
    devbranch="master",
    devurl = "latest",
)
