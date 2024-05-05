import numpy as np

def kernel(X1, X2, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    print(f"sqdist={sqdist}")
    e_sqdist = -0.5 / (l + 1e-5) ** 2 * sqdist
    print(f"e_sqdist={e_sqdist}")
    return sigma_f**2 * np.exp(e_sqdist)



k = kernel(
    np.array([-4., -3., -2., -1., 0.]).reshape(-1, 1),
    np.array([-4., -3., -2., -1., 0.]).reshape(-1, 1),
    -10, 1.0
)

print(f"k={k}")