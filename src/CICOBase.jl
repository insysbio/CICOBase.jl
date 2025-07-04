#__precompile__(false)
"""
Main module for `CICOBase.jl`.

Four functions are exported from this module for public use:

- [`get_endpoint`](@ref). Computes lower or upper endpoints of confidence interval.
- [`get_interval`](@ref). Computes confidence interval.
- [`profile`](@ref). Generates the profile function based on `loss_func`
- [`update_profile_points!`](@ref). Updates confidence interval with likelihood profile points.

"""
module CICOBase

# default values
const DEFAULT_LOCAL_ALG = :LN_NELDERMEAD
const DEFAUL_THETA_BOUNDS_ITEM = (-Inf, Inf)
const DEFAULT_LOSS_TOL = 0.

using NLopt, ForwardDiff
using Calculus
using LinearAlgebra
using RecipesBase
import PlotUtils.adapted_grid
using ProgressMeter

# include
include("structures.jl")
include("get_endpoint.jl")
include("get_interval.jl")
include("cico_one_pass.jl")
include("method_lin_extrapol.jl")
include("method_quadr_extrapol.jl")
include("profile.jl")
include("plot_interval.jl")
include("show.jl")
include("get_optimal.jl")

# export
export get_right_endpoint,
    get_endpoint,
    scaling,
    unscaling,
    ProfilePoint,
    EndPoint,
    ParamIntervalInput,
    ParamInterval,
    get_interval,
    update_profile_points!,
    get_optimal
end #module
