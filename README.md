# CICO - Confidence Intervals by Constrained Optimization

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://insysbio.github.io/CICOBase.jl/latest)
[![Build status](https://github.com/insysbio/CICOBase.jl/workflows/CI/badge.svg)](https://github.com/insysbio/CICOBase.jl/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/insysbio/CICOBase.jl/badge.svg?branch=master)](https://coveralls.io/github/insysbio/CICOBase.jl?branch=master)
[![version](https://juliahub.com/docs/General/CICOBase/stable/version.svg)](https://juliahub.com/ui/Packages/General/CICOBase)
[![GitHub license](https://img.shields.io/github/license/insysbio/CICOBase.jl.svg)](https://github.com/insysbio/CICOBase.jl/blob/master/LICENSE)
[![DOI:10.1371/journal.pcbi.1008495](https://zenodo.org/badge/DOI/10.1371/journal.pcbi.1008495.svg)](https://doi.org/10.1371/journal.pcbi.1008495)

> **Repository rename notice**  
> Up to Junuary 2025 this package was published as **LikelihoodProfiler.jl** at <https://github.com/insysbio/LikelihoodProfiler.jl>.  
> The codebase was then split: the low-level computational core was renamed **CICOBase.jl** (this repo), while the original URL now hosts a new, higher-level package—**LikelihoodProfiler.jl** (v1)—that wraps CICOBase and adds extra identifiability-analysis methods.

**CICOBase** is a [Julia language](https://julialang.org/downloads/) package for [identifiability analysis](https://en.wikipedia.org/wiki/Identifiability_analysis) and confidence intervals estimation.

See [documentation](https://insysbio.github.io/CICOBase.jl/latest/).

## Use cases

Benchmarks and use cases can be found in a separate repository: <https://github.com/insysbio/likelihoodprofiler-cases>

## Installation

```julia
julia> ]

(v1.9) pkg> add CICOBase
```

## Quick start

```julia
using CICOBase

# testing profile function
f(x) = 5.0 + (x[1]-3.0)^2 + (x[1]-x[2]-1.0)^2 + 0*x[3]^2

# Calculate parameters intervals for first parameter component, x[1]
res_1 = get_interval(
  [3., 2., 2.1], # starting point
  1,             # parameter component to analyze
  f,             # profile function
  :LIN_EXTRAPOL; # :QUADR_EXTRAPOL or :CICO_ONE_PASS
  loss_crit = 9. # critical level of loss function
  )
#

# Plot parameter profile x[1]
using Plots
plotly()
plot(res_1)
```

![Plot Linear](img/plot_lin.png?raw=true)

## License

[MIT](./LICENSE) Public license

## How to cite

**Borisov I, Metelkin E** (2020) *Confidence intervals by constrained optimization—An algorithm and software package for practical identifiability analysis in systems biology.* PLoS Comput Biol 16(12): e1008495.

Ref: <https://doi.org/10.1371/journal.pcbi.1008495>
