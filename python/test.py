import torch
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
from scipy.linalg import cho_solve, cholesky, solve_triangular

model = GaussianProcessRegressor()

x_train = np.array([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
y_train = np.array([2.0, 20.0, 2.0, 20.0]).reshape(-1, 1)

x_pred = np.array([5.0, 6.0, 7.0, 8.0]).reshape(-1, 1)

# kernel_ = C(1.0, constant_value_bounds="fixed") * RBF(
#     1.0, length_scale_bounds="fixed"
# )
#
#
# K = kernel_(x_train)
#
# print(f"K={K}")
#
# L_ = cholesky(K, lower=True, check_finite=False)
#
# print(f"L_{L_}")

model = model.fit(x_train, y_train)
mean, std  = model.predict(x_pred, return_cov=True)

print(f"mean={mean} std={std}")



# model.k
#
# t = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7])
# print(t[3:5])
# torch_sum = torch.sum(t, dim=(0))
# print(torch_sum)

# t_1 = torch.Tensor([[2, 4],[2, 4]])
# print(t_1.size())
#
# t_2 = torch.Tensor([[2],[2]])
# print(t_2.size())
#
# print(t_1/t_2)

# k = np.array([
#     [
#         [1, 11], [2, 22], [3, 33]
#     ],
#     [
#         [10, 1010], [20, 2020], [30, 3030]
#     ],
#     [
#         [100, 100100], [200, 200200], [300, 300300]
#     ],
#     [
#         [1000, 10001000], [2000, 20002000], [3000, 30003000]
#     ],
# ]
# )
#
# print(f"shape={k.shape} k={k}")
# print(f"k.T.shape={k.T.shape} k.T={k.T}")
#
#
# t_1 = torch.Tensor(k)
#
# print(f"k.T.shape={t_1.T.shape} k.T={t_1.T}")



