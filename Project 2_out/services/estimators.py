import numpy as np
from sklearn.covariance import LedoitWolf

def estimate_factor_model(asset_returns, pca_factors):
    """
    Estimates expected asset returns using PCA-transformed factor returns via OLS regression.

    Parameters:
        asset_returns (DataFrame): N x T matrix of asset returns.
        pca_factors (DataFrame): T x K matrix of PCA-transformed factor returns.

    Returns:
        mu (ndarray): Vector of expected asset returns (N,).
        betas (ndarray): Matrix of OLS regression coefficients including intercept (K+1) x N.
    """
    # Step 1: Align time indices between asset and factor returns
    common_index = asset_returns.index.intersection(pca_factors.index)
    asset_returns = asset_returns.loc[common_index]
    pca_factors = pca_factors.loc[common_index]

    # Step 2: Create design matrix X with intercept
    X = np.column_stack((np.ones(len(pca_factors)), pca_factors.values))  # (T x K+1)
    Y = asset_returns.values  # (T x N)

    # Step 3: Perform OLS regression: solve Y = Xβ
    betas = np.linalg.lstsq(X, Y, rcond=None)[0]  # (K+1 x N)
    alphas = betas[0]         # Intercept terms (N,)
    factor_betas = betas[1:]  # (K x N)

    # Step 4: Estimate expected asset returns
    factor_means = np.mean(pca_factors.values, axis=0)  # (K,)
    mu = alphas + factor_betas.T @ factor_means         # (N,)

    return mu, betas

def estimate_covariance(asset_returns, use_shrinkage=True):
    """
    Estimate the sample covariance matrix of asset returns.
    
    If use_shrinkage is True, use the Ledoit-Wolf shrinkage estimator.
    
    Parameters:
      - asset_returns: DataFrame of asset returns
      
    Returns:
      - Sigma: Covariance matrix (N x N)
    """
    if use_shrinkage:
        lw = LedoitWolf()
        lw.fit(asset_returns)
        return lw.covariance_
    else:
        return asset_returns.cov().values
