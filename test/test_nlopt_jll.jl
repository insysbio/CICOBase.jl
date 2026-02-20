using CICOBase, Test
using NLopt


f_2p(x) = 5.0 + (x[1]-3.0)^2 + (x[2]-4.0)^2  # [3., 4.]

f_2p_nt = (
    func = f_2p,
    x0 = [4.,5.],
    x1 = [3.,4.],
    endpoints = [(1.,5.),
                 (2.,6.)], 
    status = [(:BORDER_FOUND_BY_SCAN_TOL,:BORDER_FOUND_BY_SCAN_TOL),
              (:BORDER_FOUND_BY_SCAN_TOL,:BORDER_FOUND_BY_SCAN_TOL)],
    loss_crit = 9.,
    loss_optim = 5.,
    x_optim = [3., 4.]
  )


f_1p_ex(x) = 5.0 + (x[1]-1e-8)^2 # [1e-8, missing]

f_1p_ex_nt = (
    func = f_1p_ex,
    x0 = [1.5, 2.],
    x1 = [1e-8, 1.5],
    endpoints = [(-2+1e-8,2+1e-8), (nothing, nothing)], 
    status = [(:BORDER_FOUND_BY_SCAN_TOL,:BORDER_FOUND_BY_SCAN_TOL),(:SCAN_BOUND_REACHED,:SCAN_BOUND_REACHED)],
    loss_crit = 9.,
    loss_optim = 5.,
    x_optim = [1e-8, nothing]
  )

alg = :LD_TNEWTON_RESTART
  # LD_TNEWTON_PRECOND_RESTART

  f = f_2p_nt
  get_interval(
            f.x1,
            2,
            f.func,
            :CICO_ONE_PASS;
            theta_bounds=fill((-Inf,Inf),length(f.x1)),
            scan_tol=1e-6,
            local_alg = alg,
            loss_crit = f.loss_crit,
            loss_grad=:AUTODIFF,
            silent = true,
          )