var documenterSearchIndex = {"docs":
[{"location":"api/#API-references","page":"API","title":"API references","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"The package exports the following functions for parameters identifiability analysis, confidence intervals evaluation and results visualization.","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [CICOCore]\nOrder   = [:function, :type, :module]","category":"page"},{"location":"api/#CICOCore.get_endpoint","page":"API","title":"CICOCore.get_endpoint","text":"function get_endpoint(\n    theta_init::Vector{Float64},\n    scan_func::Function,\n    loss_func::Function,\n    method::Symbol,\n    direction::Symbol = :right;\n\n    loss_crit::Float64 = 0.0,\n    scale::Vector{Symbol} = fill(:direct, length(theta_init)),\n    theta_bounds::Vector{Tuple{Float64,Float64}} = unscaling.(\n        fill((-Inf, Inf), length(theta_init)),\n        scale\n        ),\n    scan_bound::Float64 = unscaling(\n        (direction==:left) ? -9.0 : 9.0,\n        scale[theta_num]\n        ),\n    scan_tol::Float64 = 1e-3,\n    loss_tol::Float64 = 0.,\n    local_alg::Symbol = :LN_NELDERMEAD,\n    kwargs...\n    )\n\nCalculates confidence interval's right or left endpoints for a function of parameters scan_func.\n\nReturn\n\nEndPoint object storing confidence interval's endpoint and intermediate profile points.\n\nArguments\n\ntheta_init: starting values of parameter vector theta. The starting values should not necessary be the optimum values of loss_func but loss_func(theta_init) should be lower than loss_crit.\nscan_func: function of parameters.\nloss_func: loss function Lambdaleft(thetaright) for profile likelihood-based (PL) identification. Usually we use log-likelihood for PL analysis: Lambda( theta ) = - 2 lnleft( L(theta) right).\nmethod: computational method to estimate confidence interval's endpoint. Currently the only supported method is: :CICO_ONE_PASS.\ndirection: :right or :left endpoint to estimate.\n\nKeyword arguments\n\nloss_crit: critical level of loss function. Confidence interval's endpoint value is the intersection point of profile likelihood and loss_crit level.\nscale: vector of scale transformations for each parameters' component. Possible values: :direct (:lin), :log, :logit. This option can speed up the optimization, especially for wide theta_bounds. The default value is :direct (no transformation) for all parameters.\ntheta_bounds: vector of tuple (lower_bound, upper_bound) for each parameter. Bounds define the ranges for possible parameter values. Default bounds are (-Inf,Inf).\nscan_bound: value which states the area of confidence point analysis.\nscan_tol: Absolute tolerance for theta_num parameter used as termination criterion.  \nloss_tol: Absolute tolerance controlling loss_func closeness to loss_crit (termination criterion). Currently doesn't work for :CICO_ONE_PASS method because of limitation in LN_AUGLAG interface.\nlocal_alg: algorithm of optimization. Derivative-free and gradient-based algorithms form NLopt package. \nmax_iter : maximal number of fitter iterations. If reaches the result status will be :MAX_ITER_STOP.\nscan_grad : For gradient optimization methods it is necessary to set how the gradient of scan_func should be calculated.\n:EMPTY (default) no gradient is set. It works only for gradient-free methods.\n:AUTODIFF means autodifferentiation from ForwardDiff package is used.\n:FINITE means finite difference method from Calculus is used.\nfunction(x::Vector{Float64}) which returns gradient vector.\nloss_grad : For gradient optimization methods it is necessary to set how the gradient of loss_func should be calculated.\n:EMPTY (default) no gradient is set. It works only for gradient-free methods.\n:AUTODIFF means autodifferentiation from ForwardDiff package is used.\n:FINITE means finite difference method from Calculus is used.\nfunction(x::Vector{Float64}) which returns gradient vector.\nsilent : Boolean argument declaring whether we display the optimization progress. Default is false\n\nSee also get_interval\n\n\n\n\n\n","category":"function"},{"location":"api/#CICOCore.get_endpoint-2","page":"API","title":"CICOCore.get_endpoint","text":"function get_endpoint(\n    theta_init::Vector{Float64},\n    theta_num::Int,\n    loss_func::Function,\n    method::Symbol,\n    direction::Symbol = :right;\n\n    loss_crit::Float64 = 0.0,\n    scale::Vector{Symbol} = fill(:direct, length(theta_init)),\n    theta_bounds::Vector{Tuple{Float64,Float64}} = unscaling.(\n        fill((-Inf, Inf), length(theta_init)),\n        scale\n        ),\n    scan_bound::Float64 = unscaling(\n        (direction==:left) ? -9.0 : 9.0,\n        scale[theta_num]\n        ),\n    scan_tol::Float64 = 1e-3,\n    loss_tol::Float64 = 0.,\n    local_alg::Symbol = :LN_NELDERMEAD,\n    max_iter::Int = 10^5,\n    loss_grad::Union{Function, Symbol} = :EMPTY,\n    silent::Bool = false\n)\n\nCalculates confidence interval's right or left endpoints for a given parameter theta_num.\n\nReturn\n\nEndPoint object storing confidence interval's endpoint and intermediate profile points.\n\nArguments\n\ntheta_init: starting values of parameter vector theta. The starting values should not necessary be the optimum values of loss_func but loss_func(theta_init) should be lower than loss_crit.\ntheta_num: index of vector component for identification: theta_init(theta_num).\nloss_func: loss function Lambdaleft(thetaright) for profile likelihood-based (PL) identification. Usually we use log-likelihood for PL analysis: Lambda( theta ) = - 2 lnleft( L(theta) right).\nmethod: computational method to estimate confidence interval's endpoint. Currently the following methods are implemented: :CICO_ONE_PASS, :LIN_EXTRAPOL, :QUADR_EXTRAPOL.\ndirection: :right or :left endpoint to estimate.\n\nKeyword arguments\n\nloss_crit : critical level of loss function. Confidence interval's endpoint value is the intersection point of profile likelihood and loss_crit level.\nscale : vector of scale transformations for each parameters' component. Possible values: :direct (:lin), :log, :logit. This option can speed up the optimization, especially for wide theta_bounds. The default value is :direct (no transformation) for all parameters.\ntheta_bounds : vector of tuple (lower_bound, upper_bound) for each parameter. Bounds define the ranges for possible parameter values. Default bounds are (-Inf,Inf).\nscan_bound : value which states the area of confidence point analysis.\nscan_tol : Absolute tolerance for theta_num parameter used as termination criterion.\nloss_tol : Absolute tolerance controlling loss_func closeness to loss_crit (termination criterion). Currently doesn't work for :CICO_ONE_PASS method because of limitation in LN_AUGLAG interface.\nlocal_alg : algorithm of optimization. Derivative-free and gradient-based algorithms form NLopt package. \nmax_iter : maximal number of fitter iterations. If reaches the result status will be :MAX_ITER_STOP.\nloss_grad : For gradient optimization methods it is necessary to set how the gradient of loss_func should be calculated.   There are options:\n:EMPTY (default) no gradient is set. It works only for gradient-free methods.\n:AUTODIFF means autodifferentiation from ForwardDiff package is used.\n:FINITE means finite difference method from Calculus is used.\nIt is also possible to set gradient function here function(x::Vector{Float64}) which returns gradient vector.\nsilent : Boolean argument declaring whether we display the optimization progress. Default is false\n\nSee also get_interval\n\n\n\n\n\n","category":"function"},{"location":"api/#CICOCore.get_interval-Tuple{Vector{Float64}, Function, Function, Symbol}","page":"API","title":"CICOCore.get_interval","text":"get_interval(\n    theta_init::Vector{Float64},\n    scan_func::Function,\n    loss_func::Function,\n    method::Symbol;\n\n    loss_crit::Float64 = 0.0,\n    scale::Vector{Symbol} = fill(:direct, length(theta_init)),\n    theta_bounds::Vector{Tuple{Float64,Float64}} = unscaling.(\n        fill((-Inf, Inf), length(theta_init)),\n        scale\n        ),\n    scan_bounds::Tuple{Float64,Float64} = unscaling.(\n        (-9.0, 9.0),\n        :direct\n        ),\n    scan_tol::Float64 = 1e-3,\n    loss_tol::Float64 = 0.,\n    local_alg::Symbol = :LN_NELDERMEAD,\n    autodiff::Bool = true,\n    kwargs...\n    )\n\nComputes confidence interval for function of parameters scan_func.\n\nReturn\n\nParamInterval structure storing all input data and estimated confidence interval.\n\nArguments\n\ntheta_init: starting values of parameter vector theta. The starting values should not necessary be the optimum values of loss_func but loss_func(theta_init) should be lower than loss_crit.\nscan_func: scan function of parameters.\nloss_func: loss function Lambdaleft(thetaright) the profile of which is analyzed. Usually we use log-likelihood for profile analysis in form Lambda( theta ) = - 2 lnleft( L(theta) right).\nmethod: computational method to estimate confidence interval's endpoint. Currently supports only :CICO_ONE_PASS method.\n\nKeyword arguments\n\nscale: vector of scale transformations for each parameters' component. Possible values: :direct (:lin), :log, :logit. This option can speed up the optimization, especially for wide theta_bounds. The default value is :direct (no transformation) for all parameters.\nscan_bounds: scan bounds tuple for scan_func values. Default is (1e-9, 1e9) .\nkwargs...: the additional arguments passed to get_endpoint\n\n\n\n\n\n","category":"method"},{"location":"api/#CICOCore.get_interval-Tuple{Vector{Float64}, Int64, Function, Symbol}","page":"API","title":"CICOCore.get_interval","text":"function get_interval(\n    theta_init::Vector{Float64},\n    theta_num::Int,\n    loss_func::Function,\n    method::Symbol;\n\n    scale::Vector{Symbol} = fill(:direct, length(theta_init)),\n    scan_bounds::Tuple{Float64,Float64} = unscaling.(\n        (-9.0, 9.0),\n        scale[theta_num]\n        ),\n    kwargs...\n)\n\nComputes confidence interval for single component theta_num of parameter vector.\n\nReturn\n\nParamInterval structure storing all input data and estimated confidence interval.\n\nArguments\n\ntheta_init: starting values of parameter vector theta. The starting values should not necessary be the optimum values of loss_func but loss_func(theta_init) should be lower than loss_crit.\ntheta_num: index of vector component for identification: theta_init(theta_num).\nloss_func: loss function Lambdaleft(thetaright) for profile likelihood-based (PL) identification. Usually we use log-likelihood for PL analysis: Lambda( theta ) = - 2 lnleft( L(theta) right).\nmethod: computational method to estimate confidence interval's endpoint. Currently the following methods are implemented: :CICO_ONE_PASS, :LIN_EXTRAPOL, :QUADR_EXTRAPOL.\n\nKeyword arguments\n\nscale: vector of scale transformations for each parameters' component. Possible values: :direct (:lin), :log, :logit. This option can speed up the optimization, especially for wide theta_bounds. The default value is :direct (no transformation) for all parameters.\nscan_bounds: scan bounds tuple for theta_num parameter. Should be within the theta_bounds for theta_num parameter. Default is (-9.,9.) for :direct scales and (1e-9, 1e+9) for :log.\nkwargs...: the additional arguments passed to get_endpoint\n\n\n\n\n\n","category":"method"},{"location":"api/#CICOCore.get_optimal-Tuple{Vector{Float64}, Function}","page":"API","title":"CICOCore.get_optimal","text":"function get_optimal(\n    theta_init::Vector{Float64}, # initial point of parameters\n    loss_func::Function; # lambda(theta)\n\n    scale::Vector{Symbol} = fill(:direct, length(theta_init)),\n    theta_bounds::Vector{Tuple{Float64,Float64}} = unscaling.(\n        fill((-Inf, Inf), length(theta_init)),\n        scale\n        ),\n    scan_tol::union{Float64,Nothing} = nothing,\n    loss_tol::Float64 = 1e-3,\n    local_alg::Symbol = :LN_NELDERMEAD,\n    silent::Bool = false,\n    max_iter::Int = 10^5,\n    loss_grad::Union{Function, Symbol} = :EMPTY\n)\n\nProvides the optimization routine using the interface similar to get_endpoint. Currently it uses standard NLopt optimization but allows to use parameter scaling and autodiff method.\n\nReturn\n\nProfilePoint object storing optimizations results.valueandlossproperties are equal to the optimal (minimal)loss_func` value.\n\nArguments\n\ntheta_init: starting values of parameter vector theta.\nloss_func: loss function Lambdaleft(thetaright) for profile likelihood-based (PL) identification. Usually we use log-likelihood for PL analysis: Lambda( theta ) = - 2 lnleft( L(theta) right).\n\nKeyword arguments\n\nscale: vector of scale transformations for each parameters' component. Possible values: :direct (:lin), :log, :logit. This option can speed up the optimization, especially for wide theta_bounds. The default value is :direct (no transformation) for all parameters.\ntheta_bounds: vector of tuple (lower_bound, upper_bound) for each parameter. Bounds define the ranges for possible parameter values. Default bounds are (-Inf,Inf).\nscan_tol: Absolute tolerance for all component of theta used as termination criterion.  \nloss_tol: Absolute tolerance controlling loss_func.\nlocal_alg: algorithm of optimization. Derivative-free and gradient-based algorithms form NLopt package.\nmax_iter : maximal number of optimizer iterations.\nloss_grad : This option declares the method for calculating loss function gradient.       This is required for gradient-based methods. The possible values       - :EMPTY (default) not gradient is set. IT works only for gradient-free methods.       - :AUTODIFF means autodifferentiation from ForwardDiff package is used.       - :FINITE means finite difference method from Calculus is used.       - It is also possible to set gradient function here function(x::Vector{Float64}) which returns gradient vector.\n\n\n\n\n\n","category":"method"},{"location":"api/#CICOCore.logistic10-Tuple{Real}","page":"API","title":"CICOCore.logistic10","text":"logistic10(x::Real)\n\nFunction transforming interval [-Inf, Inf] to [0,1] using logistic transformation. Inverse function for logit10.\n\n\n\n\n\n","category":"method"},{"location":"api/#CICOCore.logit10-Tuple{Real}","page":"API","title":"CICOCore.logit10","text":"logit10(x::Real)\n\nFunction transforming interval [0,1] to [-Inf, Inf] using logit transformation.\n\n\n\n\n\n","category":"method"},{"location":"api/#CICOCore.profile-Tuple{Vector{Float64}, Int64, Function}","page":"API","title":"CICOCore.profile","text":"function profile(\n    theta_init::Vector{Float64},\n    theta_num::Int,\n    loss_func::Function;\n\n    skip_optim::Bool = false,\n    scale::Vector{Symbol} = fill(:direct, length(theta_init)),\n    theta_bounds::Vector{Tuple{Float64,Float64}} = unscaling.(\n        fill((-Inf, Inf), length(theta_init)),\n        scale\n        ),\n    local_alg::Symbol = :LN_NELDERMEAD,\n    ftol_abs::Float64 = 1e-3,\n    maxeval::Int = 10^5,\n    kwargs... # currently not used\n)\n\nGenerates the profile function based on loss_func. Used internally in methods :LIN_EXTRAPOL, :QUADR_EXTRAPOL.\n\nReturn\n\nReturns profile function for selected parameter.\n\nArguments\n\ntheta_init: starting values of parameter vector theta. \ntheta_num: index of vector component for identification: theta_init(theta_num).\nloss_func: loss function Lambdaleft(thetaright) for profile likelihood-based (PL) identification. Usually we use log-likelihood for PL analysis: Lambda( theta ) = - 2 lnleft( L(theta) right).\n\nKeyword arguments\n\nskip_optim : set true if you need marginal profile, i.e. profile without optimization. Default is false.\nscale : \ntheta_bounds: vector of tuple (lower_bound, upper_bound) for each parameter. Bounds define the ranges for possible parameter values. Default bounds are (-Inf,Inf).\nlocal_alg: algorithm of optimization. Derivative-free and gradient-based algorithms form NLopt package.\nftol_abs : absolute tolerance criterion for profile function.\nmaxeval : maximal number of loss_func evaluations to estimate profile point.\n\n\n\n\n\n","category":"method"},{"location":"api/#CICOCore.scaling","page":"API","title":"CICOCore.scaling","text":"scaling(x::Real, scale::Symbol = :direct)\n\nTransforms values from specific scale to range [-Inf, Inf] based on option.\n\nReturn\n\nTransformed value.\n\nArguments\n\nx: input value.\nscale: transformation type: :direct (:lin), :log, :logit.\n\n\n\n\n\n","category":"function"},{"location":"api/#CICOCore.unscaling","page":"API","title":"CICOCore.unscaling","text":"unscaling(x::Real, scale::Symbol = :direct)\n\nTransforms values from [-Inf, Inf] to specific scale based on option. Inverse function for scaling.\n\nReturn\n\nTransformed value.\n\nArguments\n\nx: input value.\nscale: transformation type: :direct (:lin), :log, :logit.\n\n\n\n\n\n","category":"function"},{"location":"api/#CICOCore.update_profile_points!-Tuple{ParamInterval}","page":"API","title":"CICOCore.update_profile_points!","text":"update_profile_points!(\n    pi::ParamInterval;\n    max_recursions::Int = 2\n    )\n\nRefines profile points to make your plot more smooth. Internally uses adapted_grid to compute additional profile points. See PlotUtils.adapted_grid.\n\nArguments\n\npi : ParamInterval structure to update.\nmax_recursions : how many times each interval is allowed to be refined (default: 2).\n\n\n\n\n\n","category":"method"},{"location":"api/#RecipesBase.apply_recipe-Tuple{AbstractDict{Symbol, Any}, ParamInterval}","page":"API","title":"RecipesBase.apply_recipe","text":"using Plots\nplot(pi::ParamInterval)\n\nPlots profile L(theta) for parameter theta_num, identifiability level, identifiability interval. Use update_profile_points!(pi::ProfileInterval) function to refine profile points and make your plot more smooth\n\n\n\n\n\n","category":"method"},{"location":"api/#CICOCore.EndPoint","page":"API","title":"CICOCore.EndPoint","text":"struct EndPoint\n    value::Union{Real, Nothing}           # value of endpoint or `nothing`\n    profilePoints::Array{ProfilePoint, 1} # vector of profile points\n    status::Symbol                        # status\n    direction::Symbol                     # `:right` or `:left`\n    counter::Int                          # number of `loss_func` evaluations\n    supreme::Union{Real, Nothing}         # maximal value inside profile interval\nend\n\nStructure storing one endpoint of confidence interval.\n\nstatus values: :BORDER_FOUND_BY_SCAN_TOL, :BORDER_FOUND_BY_LOSS_TOL,  :SCAN_BOUND_REACHED, :MAX_ITER_STOP, :LOSS_ERROR_STOP\n\n\n\n\n\n","category":"type"},{"location":"api/#CICOCore.ParamInterval","page":"API","title":"CICOCore.ParamInterval","text":"struct ParamInterval\n    input::ParamIntervalInput\n    loss_init::Float64\n    method::Symbol\n    result::Tuple{EndPoint, EndPoint}\nend\n\nStructure storing result of parameter identification\n\n\n\n\n\n","category":"type"},{"location":"api/#CICOCore.ParamIntervalInput","page":"API","title":"CICOCore.ParamIntervalInput","text":"struct ParamIntervalInput <: AbstractIntervalInput\n    theta_init::Vector{Float64} # initial parameters vector\n    theta_num::Int              # number of the parameter for identifiability analysis\n    loss_func::Function         # loss function\n    method::Symbol\n    options::Any\nend\n\nStructure storing input data for parameter identification\n\n\n\n\n\n","category":"type"},{"location":"api/#CICOCore.ProfilePoint","page":"API","title":"CICOCore.ProfilePoint","text":"struct ProfilePoint\n    value::Float64               # x value of profile point\n    loss::Float64                # y value of profile point (loss function value)\n    params::Array{Float64, 1}    # vector of optimal values of `loss_func` arguments\n    ret::Symbol                  # `NLOpt.optimize()` return value \n    counter::Union{Int, Nothing} # number of `loss_func` evaluations\nend\n\nStructure storing one point from profile function. ret values: :FORCED_STOP, :MAXEVAL_REACHED, :FTOL_REACHED\n\n\n\n\n\n","category":"type"},{"location":"api/#CICOCore.CICOCore","page":"API","title":"CICOCore.CICOCore","text":"Main module for CICOCore.jl.\n\nFour functions are exported from this module for public use:\n\nget_endpoint. Computes lower or upper endpoints of confidence interval.\nget_interval. Computes confidence interval.\nprofile. Generates the profile function based on loss_func\nupdate_profile_points!. Updates confidence interval with likelihood profile points.\n\n\n\n\n\n","category":"module"},{"location":"methods/#Methods","page":"Methods","title":"Methods","text":"","category":"section"},{"location":"methods/","page":"Methods","title":"Methods","text":"Three methods implemented in CICOCore package: :CICO_ONE_PASS,  :LIN_EXTRAPOL,  :QUADR_EXTRAPOL.","category":"page"},{"location":"methods/","page":"Methods","title":"Methods","text":"The main function for CI endpoints estimation is get_interval (see API section). Estimation method is defined by the keyword argument method. It supports one of the above values. Default is :CICO_ONE_PASS.","category":"page"},{"location":"methods/#:CICO_ONE_PASS","page":"Methods","title":":CICO_ONE_PASS","text":"","category":"section"},{"location":"methods/","page":"Methods","title":"Methods","text":"The method uses the one-pass calculation of confidence interval endpoint and  utilizes Inequality-based Constrained Optimization for efficient determination of confidence intervals and detection of “non-identifiable” parameters.","category":"page"},{"location":"methods/","page":"Methods","title":"Methods","text":"The method internally calls NLopt algorithm to build an augmented objective function with LN_AUGLAG algorithm. Optimization algorithm choice is described in Optimizers section.","category":"page"},{"location":"methods/#:LIN_EXTRAPOL","page":"Methods","title":":LIN_EXTRAPOL","text":"","category":"section"},{"location":"methods/","page":"Methods","title":"Methods","text":"The method examines profile likelihood function by making steps in both directions from the optima.  Each step is derived as a linear extrapolation: y = a*x + b.","category":"page"},{"location":"methods/#:QUADR_EXTRAPOL","page":"Methods","title":":QUADR_EXTRAPOL","text":"","category":"section"},{"location":"methods/","page":"Methods","title":"Methods","text":"The method examines profile likelihood function by making steps in both directions from the optima.  Each step is derived as a quadratic extrapolation: y = x^2*a + x*b + c.","category":"page"},{"location":"visualization/#Visualization","page":"Visualization","title":"Visualization","text":"","category":"section"},{"location":"visualization/","page":"Visualization","title":"Visualization","text":"CICOCore.get_interval function returns estimated confidence interval along with other data as CICOCore.ParamInterval structure.","category":"page"},{"location":"visualization/","page":"Visualization","title":"Visualization","text":"CICOCore provides a @recipe for Plots.jl to visualize confidence interval estimation and plot parameter's profile based on CICOCore.ParamInterval.","category":"page"},{"location":"visualization/","page":"Visualization","title":"Visualization","text":"using CICOCore\n\n# Likelihood function\nf(x) = 5.0 + (x[1]-3.0)^2 + (x[1]-x[2]-1.0)^2 + 0*x[3]^2\n\n# Calculate parameters intervals for x[1], x[2], x[3]\nres = [\n    get_interval(\n        [3., 2., 2.1],\n        i,\n        f,\n        :CICO_ONE_PASS;\n        loss_crit = 9.\n    ) for i in 1:3]\n\n# Plot parameter profile x[2]\nusing Plots\nplotly()\nplot(res[2])","category":"page"},{"location":"visualization/","page":"Visualization","title":"Visualization","text":"(Image: )","category":"page"},{"location":"visualization/","page":"Visualization","title":"Visualization","text":"To make a smooth plot one can compute more profile points with CICOCore.update_profile_points! which internally uses PlotUtils.adapted_grid","category":"page"},{"location":"visualization/","page":"Visualization","title":"Visualization","text":"update_profile_points!(res[2])\n\nplot(res[2])","category":"page"},{"location":"visualization/","page":"Visualization","title":"Visualization","text":"(Image: )","category":"page"},{"location":"#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"CICOCore is a Julia package for identifiability analysis and confidence intervals estimation.","category":"page"},{"location":"#Use-cases","page":"Home","title":"Use cases","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Notebooks with use cases can be found in a separate repository: https://github.com/insysbio/likelihoodprofiler-cases","category":"page"},{"location":"","page":"Home","title":"Home","text":"Case Ref\nPK model with saturation in elimination (Image: Binder)\nLocal optim methods comparison (Image: Binder)\nTGF-β Signaling Pathway Model (Image: Binder)\nSIR Model. A simple model used as an exercise in identifiability analysis. (Image: Binder)\nCancer Taxol Treatment Model (Image: Binder)","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"julia> ]\n\n(v1.9) pkg> add CICOCore","category":"page"},{"location":"#Quick-start","page":"Home","title":"Quick start","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using CICOCore\n\n# testing profile function\nf(x) = 5.0 + (x[1]-3.0)^2 + (x[1]-x[2]-1.0)^2 + 0*x[3]^2\n\n# Calculate parameters intervals for first parameter component, x[1]\nres_1 = get_interval(\n  [3., 2., 2.1], # starting point\n  1,             # index of parameter\n  f,             # profile function\n  :LIN_EXTRAPOL; # method\n  loss_crit = 9. # likelihood confidence level\n  )\n#\n\n# Plot parameter profile x[1]\nusing Plots\nplotly()\nplot(res_1)","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: plot_lin)","category":"page"},{"location":"#Intro","page":"Home","title":"Intro","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The reliability and predictability of a kinetic systems biology (SB) and systems pharmacology (SP) model depends on the calibration of model parameters. Experimental data can be insufficient to determine all the parameters unambiguously. This results in \"non-identifiable\" parameters and parameters identifiable within confidence intervals (CIs). The algorithms included in CICOCore implement Profile Likelihood (PL) [2] method for parameters identification and can be applied to complex SB models. The results of the algorithms can be used to qualify and calibrate parameters or to reduce the model.","category":"page"},{"location":"#Objective","page":"Home","title":"Objective","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The package introduces several original algorithms. The default algorithm :CICO_ONE_PASS has been developed in accordance with the following principles:","category":"page"},{"location":"","page":"Home","title":"Home","text":"The algorithms don't require the likelihood function to be differentiable. Hence, derivative-free or global optimization methods can be used to estimate CI endpoints.\nThe algorithms are designed to obtain CI endpoints and avoid the calculation of profiles as the most computationally expensive part of the analysis. \nCI endpoints are estimates with some preset tolerance. Reasonable tolerance setup can also reduce the number of likelihood function calls and speed up the computations. \nThe algorithm are applicable for both finite and infinite CI.","category":"page"},{"location":"#Methods-overview","page":"Home","title":"Methods overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This algorithms can be applied to complex kinetic models where function differentiability is not guaranteed and each likelihood estimation is computationally expensive.  ","category":"page"},{"location":"","page":"Home","title":"Home","text":"The package introduces original \"one-pass\" algorithm: Confidence Intervals evaluation by Constrained Optimization [6]  :CICO_ONE_PASS developed by the authors of this package. :CICO_ONE_PASS utilizes the Inequality-based Constrained Optimization [3-4] for efficient determination of confidence intervals and detection of “non-identifiable” parameters.  ","category":"page"},{"location":"","page":"Home","title":"Home","text":"The \"multi-pass\" methods use extrapolation/interpolation of profile likelihood points: linear (:LIN_EXTRAPOL) and quadratic (:QUADR_EXTRAPOL) approaches. They are also efficient for both identifiable and non-identifiable parameters.","category":"page"},{"location":"#References","page":"Home","title":"References","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Wikipedia Identifiability_analysis\nKreutz C., et al. Profile Likelihood in Systems Biology. FEBS Journal 280(11), 2564-2571, 2013\nSteven G. Johnson, The NLopt nonlinear-optimization package, link\nAndrew R. Conn, Nicholas I. M. Gould, and Philippe L. Toint, \"A globally convergent augmented Lagrangian algorithm for optimization with general constraints and simple bounds,\" SIAM J. Numer. Anal. vol. 28, no. 2, p. 545-572 (1991)\nJulia: A Fresh Approach to Numerical Computing. Jeff Bezanson, Alan Edelman, Stefan Karpinski and Viral B. Shah (2017) SIAM Review, 59: 65–98\nBorisov I., Metelkin E. An Algorithm for Practical Identifiability Analysis and Confidence Intervals Evaluation Based on Constrained Optimization. 2018. October. ICSB2018. https://doi.org/10.13140/RG.2.2.18935.06563","category":"page"},{"location":"optimizers/#Note-on-optimizers","page":"Optimizers","title":"Note on optimizers","text":"","category":"section"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"Currently CICOCore relies on NLopt. ","category":"page"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"Internally CICOCore utilizes :LN_AUGLAG algorithm from NLopt to construct an augmented objective function with a set of bound constrains. Then the augmented objective function is passed to an optimization algorithm, which is defined by the keyword argument local_alg (see CICOCore.get_interval).","category":"page"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"Default local_alg = :LN_NELDERMEAD, which is the most reliable for the current problem among the derivative-free algorithms.  Any NLopt algorithm could be potentially used as local_alg. Nevertheless we performed a series of simulation experiments displaying that some of algorithms may fail at some specific problems.","category":"page"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"The following algorithms have shown satisfactory results on test problems (see also /test/test_grad_algs.jl):","category":"page"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":":LN_NELDERMEAD\n:LD_MMA\n:LD_SLSQP\n:LD_CCSAQ","category":"page"}]
}
