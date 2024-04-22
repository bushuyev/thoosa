#https://krasserm.github.io/2018/03/19/gaussian-processes/

import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from python.gaussian_processes_util import plot_gp


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
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


def posterior(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    """
    Computes the suffifient statistics of the posterior distribution
    from m training data X_train and Y_train and n new inputs X_s.

    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.

    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    """
    print(f"X_s={X_s}")
    print(f"X_train={X_train}")
    print(f"Y_train={Y_train}")

    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    print(f"K={K}")
    K_s = kernel(X_train, X_s, l, sigma_f)
    print(f"K_s={K_s}")
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))

    print(f"K_ss={K_ss}")

    K_inv = inv(K)

    print(f"K_inv={K_inv}")

    # Equation (7)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (8)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s

X = np.arange(-5, 5, 0.5).reshape(-1, 1)


# Mean and covariance of the prior
# mu = np.zeros(X.shape)
# cov = kernel(X, X)
#
# # Draw three samples from the prior
# samples = np.random.multivariate_normal(mu.ravel(), cov, 3);
#
# plot_gp(mu, cov, X, samples=samples)
#
# plt.show()


X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
Y_train = np.sin(X_train)

# print(f"Y_train={X_train**2}")
# print(f"X_train={X_train}")
# print(f"Y_train={Y_train}")
# print(f"kernel={kernel(X_train, Y_train, 1.0, 1.0)}")

# Compute mean and covariance of the posterior distribution
mu_s, cov_s = posterior(X, X_train, Y_train)

con_t = torch.tensor(cov_s).float()
chol = con_t.float().cholesky()
print(f"choltype={chol.type()} chol={chol}")

print(f"mu_s={mu_s.shape} cov_s={cov_s.shape}")
print(f"mu_s={mu_s}")
print(f"cov_s={cov_s.flatten()}")

# samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
random = [np.random.normal() for x in range(20)]
mu_t = torch.tensor(mu_s).float()
random_t = torch.tensor(random).float()
samples = [
    random,
    (mu_t.view(20, 1) + torch.matmul(chol, random_t).view(20, 1)).numpy(),
    (mu_t.view(20, 1) + torch.matmul(con_t, random_t).view(20, 1)).numpy()
]

plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples)
plt.show()