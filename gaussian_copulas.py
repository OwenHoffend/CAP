import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal
from statsmodels.distributions.copula.api import GaussianCopula
from CAP import scc

def get_Gaussian_copula_func(Cin, num_restarts=100):
    """
    Parameters
    ----------
    Cin : np.ndarray
        Target SCC matrix.
    Px : np.ndarray
        Marginal probabilities P(X_i = 1).
    num_restarts : int
        Number of random initializations.

    Returns
    -------
    GC: A Gaussian copula function that can be called
    """


    """First fit a Gaussian copula to the desired correlation and marginals"""
    n = Cin.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def xvec_to_Pearson(x):
        B = x.reshape(n, n)

        norms = np.linalg.norm(B, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)

        B /= norms
        R = B @ B.T
        return R

    def GC(Px):

        def objective(x):
            R = xvec_to_Pearson(x)

            err = 0.0
            for i, j in pairs:

                pi, pj = Px[i], Px[j]
                rho = np.clip(R[i, j], -0.999999, 0.999999)

                ti = norm.ppf(pi)
                tj = norm.ppf(pj)

                cov = np.array([
                    [1.0, rho],
                    [rho, 1.0],
                ])

                p_ij = multivariate_normal.cdf(
                    [ti, tj],
                    mean=[0.0, 0.0],
                    cov=cov,
                )

                c_hat = scc(pi, pj, p_ij)
                diff = c_hat - Cin[i, j]
                err += diff * diff

            return err

        best_result = None
        best_value = np.inf

        rng = np.random.default_rng(0)

        for _ in range(num_restarts):
            x0 = rng.normal(size=(n, n)).ravel()

            result = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                options={
                    "maxiter": 1000,
                    "ftol": 1e-12,
                },
            )

            if result.fun < best_value:
                best_result = result
                best_value = result.fun

        R_best = xvec_to_Pearson(best_result.x, n, n)
        R_best = 0.5 * (R_best + R_best.T) #FIXME: not sure what this line does
        np.fill_diagonal(R_best, 1.0)
        return GaussianCopula(R_best, allow_singular=True).cdf

    return GC