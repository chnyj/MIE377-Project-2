# factor_model_selection.py

from sklearn.decomposition import PCA
import numpy as np

def apply_pca_to_factors(factors, n_components):
    """
    Applies Principal Component Analysis (PCA) to reduce the dimensionality
    of the input factor dataset.

    Parameters:
        factors (array-like): A 2D data matrix of shape (n_samples, n_features)
                              where rows are time periods and columns are factors.
        n_components (int): The number of principal components to retain.

    Returns:
        numpy.ndarray: PCA-transformed matrix of shape (n_samples, n_components),
                       representing the top principal components.
    """
    # Ensure compatibility by converting to NumPy array
    factors = np.asarray(factors)

    # Apply PCA to extract the top 'n_components' principal directions
    pca = PCA(n_components=n_components)
    transformed_factors = pca.fit_transform(factors)

    return transformed_factors

# Optional: Local testing block for script-based development
if __name__ == "__main__":
    # Example usage for quick testing/debugging
    # Generate random data: 100 samples of 5 factors
    sample_factors = np.random.rand(100, 5)

    # Apply PCA to reduce to 2 components
    reduced = apply_pca_to_factors(sample_factors, n_components=2)
    
    print("Transformed factors shape:", reduced.shape)
    print(reduced)
