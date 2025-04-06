import numpy as np
import matplotlib.pyplot as plt
from services.estimators import estimate_covariance, estimate_factor_model
from services.factor_model_selection import apply_pca_to_factors
from services.strategies import select_best_candidate
from services.strategies import ensemble_candidate

# Tracks portfolio weights across function calls if using ensemble_candidate directly
previous_weights = None

def run_trading_algorithm(prices, factors, rebalance_period=6, calibration_years=5, turnover_weight=0.01):
    """
    Executes a trading algorithm over a dataset using a rolling calibration window,
    strategy selection, and performance tracking.

    Parameters:
        prices (DataFrame): Asset price data
        factors (DataFrame): Factor return data
        rebalance_period (int): Rebalance frequency in months
        calibration_years (int): Calibration window size in years
        turnover_weight (float): Penalty for turnover in strategy selection

    Returns:
        dict: Dictionary containing returns, weights, turnover, Sharpe ratio, and strategy choices
    """
    # Convert price to monthly returns
    returns = prices.pct_change().dropna()
    num_months = returns.shape[0]
    calibration_window = calibration_years * 12  # convert years to months

    # Containers for tracking metrics and decisions
    portfolio_weights = []
    portfolio_returns = []
    turnover_list = []
    strategy_choices = []
    rolling_sharpes = []

    # Iterate through the dataset by rebalance intervals
    for t in range(calibration_window, num_months - rebalance_period + 1, rebalance_period):
        # Split historical (calibration) and future (test) windows
        calibration_returns = returns.iloc[t - calibration_window:t]
        future_returns = returns.iloc[t:t + rebalance_period]
        factor_window = factors.iloc[t - calibration_window:t]

        # Apply PCA on the calibration factor window
        pca_factors, _, _ = apply_pca_to_factors(factor_window)
        recent_market = factor_window['Mkt_RF']  # Unused here but can support CAPM-style analysis

        # Get weights from previous rebalance for turnover calculation
        prev_weights = portfolio_weights[-1] if portfolio_weights else None

        # Select the best optimizer among Sharpe, CVaR, Risk Parity
        weights, chosen_strategy, score_dict = select_best_candidate(
            calibration_returns, pca_factors, prev_weights, turnover_weight=turnover_weight
        )

        # Store chosen portfolio and metadata
        portfolio_weights.append(weights)
        strategy_choices.append(chosen_strategy)

        # Compute out-of-sample returns over the next rebalance period
        future_period = future_returns.values @ weights
        portfolio_returns.extend(future_period)

        # Calculate turnover between current and previous weights
        if prev_weights is not None:
            turnover = np.sum(np.abs(weights - prev_weights))
        else:
            turnover = 0
        turnover_list.append(turnover)

        # Calculate rolling 12-month Sharpe ratio
        if len(portfolio_returns) >= 12:
            rolling_window = portfolio_returns[-12:]
            rolling_sharpe = np.mean(rolling_window) / np.std(rolling_window)
        else:
            rolling_sharpe = np.nan
        rolling_sharpes.append(rolling_sharpe)

        print(f"Rebalance at t={t}: Selected {chosen_strategy} | Scores = {score_dict}")

    # Final performance metrics
    port_returns = np.array(portfolio_returns)
    sharpe_ratio = np.mean(port_returns) / np.std(port_returns)
    avg_turnover = np.mean(turnover_list)

    # === Plot 1: Wealth Evolution ===
    wealth = np.cumprod(1 + port_returns)
    plt.figure(figsize=(10, 4))
    plt.plot(wealth)
    plt.title("Wealth Evolution Over Time")
    plt.xlabel("Months")
    plt.ylabel("Cumulative Wealth")
    plt.grid()
    plt.show()

    # === Plot 2: Rolling Sharpe Ratio ===
    plt.figure(figsize=(10, 4))
    plt.plot(rolling_sharpes)
    plt.title("Rolling 12-Month Sharpe Ratio")
    plt.xlabel("Rebalancing Period")
    plt.ylabel("Sharpe Ratio")
    plt.grid()
    plt.show()

    return {
        'returns': port_returns,
        'weights': portfolio_weights,
        'turnover': avg_turnover,
        'sharpe_ratio': sharpe_ratio,
        'strategies': strategy_choices
    }

def project_function(prices, factors):
    """
    Alternative entry point using ensemble_candidate, which returns a blended portfolio
    based on multiple optimization strategies.

    Returns:
        ndarray: Combined weight vector from ensemble strategy
    """
    global previous_weights

    # Generate ensemble candidate weights and track their scores
    w_combined, candidate_weights, scores = ensemble_candidate(prices, factors, prev_weights=previous_weights)
    print("Ensemble candidate weights:", candidate_weights)
    print("Candidate scores:", scores)

    # Update previous weights to be used in the next call
    previous_weights = w_combined
    return w_combined
