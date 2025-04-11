import numpy as np
from services.optimization import *
from services.estimators import estimate_factor_model, estimate_covariance

# --------------------------------------------------------
# Evaluate risk-adjusted return of a portfolio
# --------------------------------------------------------

def evaluate_sharpe(weights, returns):
    """
    Computes the Sharpe ratio of a portfolio.

    Parameters:
        weights (ndarray): Portfolio weights (N,)
        returns (ndarray): Historical returns matrix (T x N)

    Returns:
        float: Sharpe ratio (mean return / standard deviation)
    """
    port_returns = returns @ weights
    return np.mean(port_returns) / (np.std(port_returns) + 1e-8)


# --------------------------------------------------------
# Select the best strategy from Sharpe, CVaR, and Risk Parity
# --------------------------------------------------------

def select_best_candidate(asset_returns, factor_returns, prev_weights=None, turnover_weight=1, alpha=0.95):
    """
    Selects the single best optimization strategy (out of Sharpe, CVaR, Risk Parity)
    based on a composite score balancing Sharpe ratio and turnover penalty.

    Parameters:
        asset_returns (DataFrame): Historical returns (T x N)
        factor_returns (DataFrame): Factor return matrix (T x K)
        prev_weights (ndarray): Previous portfolio weights (optional)
        turnover_weight (float): Penalty factor for turnover
        alpha (float): Confidence level for CVaR

    Returns:
        tuple: (best portfolio weights, strategy name, score dictionary)
    """
    Sigma = estimate_covariance(asset_returns)
    mu, _ = estimate_factor_model(asset_returns, factor_returns)

    # Generate portfolios
    weights_sharpe = optimize_sharpe(mu, Sigma)
    weights_cvar = optimize_cvar(asset_returns.values, alpha)
    weights_rp = optimize_risk_parity(Sigma)
    weights_rmvo = robust_mvo_ellipsoid(asset_returns)
    weights_rrp = robust_risk_parity(Sigma, mu)

    candidates = {
        'Sharpe': weights_sharpe,
        'CVaR': weights_cvar,
        'RiskParity': weights_rp,
        'Robust Risk Parity': weights_rrp,
        'Robust MVO': weights_rmvo
    }

    # Score each candidate
    scores = {}
    for name, w in candidates.items():
        sharpe = evaluate_sharpe(w, asset_returns.values)
        turnover_penalty = turnover_weight * np.sum(np.abs(w - prev_weights)) if prev_weights is not None else 0
        score = 0.8 * sharpe - 0.2 * (turnover_penalty ** 2)
        scores[name] = score

    # Select the strategy with the highest adjusted score
    best_strategy = max(scores, key=scores.get)
    print("BEST STRATEGY: ", best_strategy)
    return candidates[best_strategy], best_strategy, scores


# --------------------------------------------------------
# Ensemble approach combining all four strategies
# --------------------------------------------------------

def ensemble_candidate(asset_returns, factor_returns, prev_weights=None, 
                       turnover_weight=1, alpha=0.95, tau=1000,
                       reg_lambda=0.01, transaction_cost_weight=0.01):
    """
    Creates an ensemble portfolio by combining Sharpe, CVaR, and Risk Parity strategies,
    weighted by a softmax-transformed score.

    Parameters:
        asset_returns (DataFrame): Historical returns (T x N)
        factor_returns (DataFrame): Factor model input
        prev_weights (ndarray): Previous weights for turnover penalty
        turnover_weight (float): Multiplier for turnover penalty
        alpha (float): CVaR confidence level
        tau (float): Temperature for softmax (controls spread of weights)
        reg_lambda (float): L1 regularization in Sharpe optimizer
        transaction_cost_weight (float): Weight for penalizing portfolio change

    Returns:
        tuple: (combined ensemble weights, candidate softmax weights, raw scores)
    """
    # Estimate expected returns and risk
    Sigma = estimate_covariance(asset_returns, use_shrinkage=True)
    mu, _ = estimate_factor_model(asset_returns, factor_returns)

    # Run all three strategies with penalties
    w_sharpe = optimize_sharpe(mu, Sigma, reg_lambda=reg_lambda, 
                               prev_weights=prev_weights, 
                               transaction_cost_weight=transaction_cost_weight)
    w_cvar = optimize_cvar(asset_returns.values, alpha, 
                           prev_weights=prev_weights, 
                           transaction_cost_weight=transaction_cost_weight)
    w_rp = optimize_risk_parity(Sigma, prev_weights=prev_weights, 
                                transaction_cost_weight=transaction_cost_weight)
    w_rmvo = robust_mvo_ellipsoid(asset_returns)
    w_rrp = robust_risk_parity(Sigma, mu)

    candidates = { "Robust Risk Parity": w_rrp, "Cvar": w_cvar}

    # Score each strategy with turnover penalty
    scores = {}
    for name, w in candidates.items():
        sharpe = evaluate_sharpe(w, asset_returns.values)
        turnover = np.sum(np.abs(w - prev_weights)) if prev_weights is not None else 0
        turnover_norm = turnover / np.sqrt(len(w))
        scores[name] = 0.8 * sharpe - 0.2 * (turnover_norm ** 2) * turnover_weight # 80-20 split for sharpe and turnover 

    # Convert scores into softmax weights for ensemble combination
    score_array = np.array(list(scores.values()))
    exp_scores = np.exp(score_array / tau)
    candidate_weights = exp_scores / np.sum(exp_scores)

    # Final ensemble portfolio (weighted combination of all strategies)
    w_combined = (
                #   
                  candidate_weights[0] * w_cvar +
                  candidate_weights[1] * w_rrp 
                #   candidate_weights[2] * w_rmvo 
                #   candidate_weights[2] * w_sharpe
                #   candidate_weights[2] * w_rp
                  
                  
                  )
    # print("candidates weights sum: ", sum(candidate_weights))

    return w_combined, candidate_weights, scores
