import numpy as np
import torch

# def kernel(X1, X2, l=1.0, sigma_f=1.0):
#     """
#     Isotropic squared exponential kernel.
#
#     Args:
#         X1: Array of m points (m x d).
#         X2: Array of n points (n x d).
#
#     Returns:
#         (m x n) matrix.
#     """
#     sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
#     print(f"sqdist={sqdist}")
#     return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)
#
#
# x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
#
# print(kernel(x, x, sigma_f=1.0))
dist = torch.cdist(torch.Tensor([1,2,3,4,5]).view(-1,1), torch.Tensor([1,2,3,4,5]).view(-1,1), 2)
print(f"dist={dist}")