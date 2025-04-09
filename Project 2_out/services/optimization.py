import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

# -------------------------------
# Mean-Variance Optimization (MVO)
# -------------------------------

def MVO(mu, Q):
    """
    Basic Mean-Variance Optimization using convex quadratic programming (CVXPY).

    Parameters:
        mu (ndarray): Expected returns (N,)
        Q (ndarray): Covariance matrix (N x N)

    Returns:
        ndarray: Optimal portfolio weights (N,)
    """
    n = len(mu)
    targetRet = np.mean(mu)
    lb = np.zeros(n)

    A = -1 * mu.T
    b = -1 * targetRet

    Aeq = np.ones((1, n))
    beq = 1

    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q)),
                      [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb])
    prob.solve()
    return x.value


# --------------------------------------
# Robust MVO with Ellipsoid Uncertainty 
# --------------------------------------

def robust_mvo_ellipsoid(rets, lam=100, rho=0.05):
    """
    Robust Mean-Variance Optimization under ellipsoidal uncertainty in expected returns.
    
    Parameters:
    - rets: T x N matrix of returns (T: time, N: assets)
    - lam: risk aversion parameter (higher = more risk-averse)
    - rho: confidence radius for ellipsoidal uncertainty (larger = more conservative)
    
    Returns:
    - Optimal weights (N x 1)
    """

    T, N = rets.shape

    # Estimate mean and covariance from data
    mu_hat = np.mean(rets, axis=0)
    Sigma = np.cov(rets.T)         # Asset return covariance
    Sigma_mu = np.cov(rets.T) / T  # Estimation covariance of the mean

    # Objective function: risk-adjusted robust return
    def objective(w):
        # Nominal mean-variance utility
        mv_obj = lam * np.dot(w, Sigma @ w) - np.dot(w, mu_hat)
        # Robust penalty (ellipsoidal): w^T Σ_μ w
        penalty = rho * np.sqrt(np.dot(w, Sigma_mu @ w))
        return mv_obj + penalty

    # Constraints: weights sum to 1, no short selling
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(N)]
    w0 = np.ones(N) / N  # Initial guess

    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed: " + result.message)



# -------------------------------
# Enhanced Sharpe Ratio Optimization
# -------------------------------

def optimize_sharpe(mu, Sigma, reg_lambda=0.01, prev_weights=None, transaction_cost_weight=0.01):
    """
    Sharpe ratio maximization with optional L1 (LASSO) regularization and transaction cost penalty.

    Parameters:
        mu (ndarray): Expected returns (N,)
        Sigma (ndarray): Covariance matrix (N x N)
        reg_lambda (float): L1 penalty weight
        prev_weights (ndarray or None): Portfolio weights from previous period (for turnover penalty)
        transaction_cost_weight (float): Penalty weight for portfolio changes

    Returns:
        ndarray: Optimal portfolio weights (N,)
    """
    n = len(mu)
    init_weights = np.ones(n) / n

    def neg_sharpe(w):
        port_return = np.dot(mu, w)
        port_vol = np.sqrt(np.dot(w.T, Sigma @ w))
        sharpe = port_return / (port_vol + 1e-8)

        reg_penalty = reg_lambda * np.sum(np.abs(w))  # L1 regularization
        tc_penalty = 0
        if prev_weights is not None:
            tc_penalty = transaction_cost_weight * np.sum((w - prev_weights) ** 2)

        return -(sharpe - reg_penalty - tc_penalty)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(n)]

    result = minimize(neg_sharpe, init_weights, bounds=bounds, constraints=constraints)
    return result.x


# -------------------------------
# CVaR Optimization (Conditional Value at Risk)
# -------------------------------

def optimize_cvar(returns, alpha=0.95, prev_weights=None, transaction_cost_weight=0.01):
    """
    CVaR portfolio optimization using linear programming via CVXPY.

    Parameters:
        returns (ndarray): Historical returns matrix (T x N)
        alpha (float): CVaR confidence level (e.g., 0.95)
        prev_weights (ndarray or None): Previous period weights
        transaction_cost_weight (float): Turnover penalty weight

    Returns:
        ndarray: Optimal portfolio weights (N,)
    """
    T, N = returns.shape
    w = cp.Variable(N)
    VaR = cp.Variable()
    z = cp.Variable(T)

    loss = -returns @ w
    tc_penalty = 0
    if prev_weights is not None:
        tc_penalty = transaction_cost_weight * cp.sum_squares(w - prev_weights)

    objective = cp.Minimize(VaR + (1 / ((1 - alpha) * T)) * cp.sum(z) + tc_penalty)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        z >= 0,
        z >= loss - VaR
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()
    return w.value


# -------------------------------
# Risk Parity Optimization
# -------------------------------

def optimize_risk_parity(Sigma, prev_weights=None, transaction_cost_weight=0.01):
    """
    Risk parity optimization: equalizes the risk contribution of each asset.

    Parameters:
        Sigma (ndarray): Covariance matrix (N x N)
        prev_weights (ndarray or None): Previous weights for turnover penalty
        transaction_cost_weight (float): Penalty for deviation from previous weights

    Returns:
        ndarray: Optimal portfolio weights (N,)
    """
    N = Sigma.shape[0]
    init_weights = np.ones(N) / N

    def risk_contribution(w):
        portfolio_vol = np.sqrt(w.T @ Sigma @ w)
        marginal = Sigma @ w
        contrib = w * marginal
        return contrib / (portfolio_vol + 1e-8)

    def objective(w):
        rc = risk_contribution(w)
        base_obj = np.sum((rc - np.mean(rc)) ** 2)
        if prev_weights is not None:
            base_obj += transaction_cost_weight * np.sum((w - prev_weights) ** 2)
        return base_obj

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(N)]

    result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


# -------------------------------
# Robust Risk Parity Optimization
# -------------------------------

def robust_risk_parity(Q_hat, mu, c = 1, rho=0.1):

    n = len(mu)

    # Variables
    y = cp.Variable(n, pos=True)

    # Worst-case quadratic term: maximize y^T (Q_hat + Δ) y s.t. ||Δ||_F ≤ rho
    # This max evaluates to: y^T Q_hat y + rho * ||yy^T||_F = y^T Q_hat y + rho * ||y||_2^2
    robust_quadratic_term = cp.quad_form(y, Q_hat) + rho * cp.norm(y, 2)**2

    # Objective: minimize worst-case risk - log barrier
    objective = cp.Minimize(0.5 * robust_quadratic_term - c * cp.sum(cp.log(y)))
    prob = cp.Problem(objective)
    prob.solve()

    # Normalize to get asset weights
    y_star = y.value
    x_star = y_star / np.sum(y_star)

    return x_star