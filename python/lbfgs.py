import math

import torch
from math import log
from torch import optim
from matplotlib import pyplot
from torch.linalg import cholesky, solve_triangular

def kernel(t_0:torch.Tensor, t_1:torch.Tensor, l:torch.Tensor, sigma_f:torch.Tensor):
    """
    Isotropic squared exponential kernel.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    # sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    # return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

    return (torch.cdist(t_0, t_1, 2.0) ** 2).multiply((-0.5 * l ** 2)).exp().multiply(sigma_f ** 2)


def nll_fn(X_train, Y_train, noise, naive=True):
    Y_train = Y_train.ravel()

    def nll_stable(theta):
        # Numerically more stable implementation of Eq. (11) as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.
        noise = 0.1
        print(f"theta={theta}")

        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + noise**2 * torch.zeros(X_train.size()).fill_diagonal_(noise, False)

        print(f"K={K}")

        L:torch.Tensor = cholesky(K)

        print(f"L={L}")
        print(f"L.T={L.T}")

        S1 = solve_triangular(L, Y_train, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)

        print(f"S1={S1}")
        print(f"S2={S2}")

        diagonal:torch.Tensor = L.diagonal()
        print("diagonal={:?}", diagonal)

        log_t = diagonal.log()
        print("log={:?}", log)

        np_sum = log_t.sum()
        print(f"np_sum={np_sum}")

        dot = Y_train.dot(S2)

        print(f"dot={dot}")

        v = np_sum + 0.5 * dot + 0.5 * len(X_train) * log(2 * math.pi)

        print(f"v={v}")
        return v

    return nll_stable

# Define the parabola function
def func(x:torch.Tensor):
    v = (x - 20) ** 2 + (x).sin()
    print(f"func: x={x} v={v}")
    return v
    # x_0 = (x-50)/10
    # return  x_0 ** 2 + 100 * x_0.sin()

X_train = torch.tensor([-4, 1, 1.0]).view(-1, 1)
Y_train = X_train.sin()


# Initialize the variable to optimize (parameter)
x = torch.tensor([1.0, 1.0], requires_grad=True)

# Define the LBFGS optimizer
optimizer = optim.LBFGS([x], line_search_fn="strong_wolfe")#, lr=0.01, max_iter=1000, line_search_fn='strong_wolfe')

# Closure to evaluate the function and its gradient
def closure():
    optimizer.zero_grad()  # Zero the gradients
    # output = func(x)   # Evaluate the function
    output = nll_fn(X_train, Y_train, 0.1)(x)   # Evaluate the function
    output.backward()      # Compute gradients
    return output

# Run optimization
optimizer.step(closure)  # Take one optimization step
# max_iter = 10
# for i in range(max_iter):
#     optimizer.step(closure)  # Take one optimization step
#     print(f"Iteration {i+1}: x = {x.item()}")
    # print(f"Iteration {i+1}: x = {x.item()}, f(x) = {parabola(x).item()}")

print("Optimization finished.")
print(f"Minimum found at x = {x.item()}, f(x) = {func(x).item()}")

# pyplot.plot(range(-100,100), [func(torch.tensor(x)) for x in range(-100, 100)], 'r')
# pyplot.show()
