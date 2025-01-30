
using CICO
using Documenter

makedocs(
    modules = [CICO],
    sitename = "CICO Documentation",
    pages = [
        "Home" => "index.md",
        "Methods" => "methods.md",
        "Visualization" => "visualization.md",
        "Optimizers" => "optimizers.md",
        "API" => "api.md",
    ],
)


deploydocs(
    repo   = "github.com/insysbio/CICO.jl.git"
)
