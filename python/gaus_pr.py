#https://krasserm.github.io/2018/03/19/gaussian-processes/

import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from python.gaussian_processes_util import plot_gp
from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

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
    return sigma_f**2 * np.exp(-0.5 / (l+1e-5)**2 * sqdist)


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




def nll_fn(X_train, Y_train, noise, naive=True):
    Y_train = Y_train.ravel()

    def nll_stable(theta):
        # Numerically more stable implementation of Eq. (11) as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.
        # noise = 0.1
        # print(f"theta={theta}")

        # K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + noise**2 * np.eye(len(X_train))
        l = theta[0]
        sigma_f = 1.0
        print(f"l={l} sigma_f={sigma_f}")

        noise_t = noise ** 2 * np.eye(len(X_train))
        print(f"noise_t={noise_t}")

        K = kernel(X_train, X_train, l=l, sigma_f=sigma_f) + noise_t

        print(f"K={K}")

        L = cholesky(K)

        print(f"L={L}")
        print(f"L.T={L.T}")

        S1 = solve_triangular(L, Y_train, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)

        print(f"S1={S1}")
        print(f"S2={S2}")

        diagonal = np.diagonal(L)
        print("diagonal={:?}", diagonal)

        log = np.log(diagonal)
        print("log={:?}", log)

        np_sum = np.sum(log)
        print(f"np_sum={np_sum}")

        dot = Y_train.dot(S2)

        print(f"dot={dot}")

        v = np_sum + 0.5 * dot + 0.5 * len(X_train) * np.log(2 * np.pi)

        print(f"v={v}")
        return v

    return nll_stable

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
# X_train = np.arange(-4, 1, 1.0).reshape(-1, 1)
# Y_train = np.sin(X_train)
Y_train = np.array([0.7568025, -0.14112001, -0.90929743, -0.84147098, 0.]).reshape(-1, 1)

# print(f"Y_train={X_train**2}")
# print(f"X_train={X_train}")
# print(f"Y_train={Y_train}")
# print(f"kernel={kernel(X_train, Y_train, 1.0, 1.0)}")

# Compute mean and covariance of the posterior distribution
# mu_s, cov_s = posterior(X, X_train, Y_train, sigma_f=2.0)

# con_t = torch.tensor(cov_s).float()
# chol = con_t.float().cholesky()

# print(f"choltype={chol.type()} chol={chol}")
#
# print(f"mu_s={mu_s.shape} cov_s={cov_s.shape}")
# print(f"mu_s={mu_s}")
# print(f"cov_s={cov_s.flatten()}")

# samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)

#---
# random = [np.random.normal() for x in range(20)]
# mu_t = torch.tensor(mu_s).float()
# random_t = torch.tensor(random).float()
# samples = [
#     random,
#     (mu_t.view(20, 1) + torch.matmul(chol, random_t).view(20, 1)).numpy(),
#     (mu_t.view(20, 1) + torch.matmul(con_t, random_t).view(20, 1)).numpy()
# ]
#
# plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples)
#---

#---
print(f"X_train={X_train} Y_train={Y_train}")

res = minimize(nll_fn(X_train, Y_train, 0.1), np.array([10.0]),
               # bounds=((1e-5, None), (1e-5, None)),
               method='BFGS')

# l_opt, sigma_f_opt = res.x
l_opt = res.x
sigma_f_opt = 1.0

print(f"l_opt={l_opt} sigma_f_opt={sigma_f_opt}")
#---

# Compute posterior mean and covariance with optimized kernel parameters and plot the results
# mu_s, cov_s = posterior(X, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=0.1)
# plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)
#
# x = list(map(lambda x: x / 10., range(-100, 100)))
# plt.plot(x, [nll_fn(X_train, Y_train, 0.1)([x]) for x in x], label='Mean')
# plt.show()