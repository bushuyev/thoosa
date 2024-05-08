from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

def func(x):
    x_ = (x - 10) ** 2
    return x_ + 2 * np.sin(x_)

if __name__ == "__main__":
    print("ok")

#'Newton-CG', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'
# for m in ['Nelder-Mead', 'Powell', 'CG', 'BFGS',  'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr',  ]:
m = 'Powell'
res = [minimize(func, np.array([x_0]),
               # bounds=((1e-5, None), (1e-5, None)),
               method=m).x for x_0 in [-1000, -10, 0, 10, 1000] ]

print(f"m={m} res={res}")

# x = np.arange(0, 20, step=0.1)
# plt.plot(x, [func(x) for x in x], label='Mean')
# plt.show()
