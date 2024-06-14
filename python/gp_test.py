import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

# k1 = kernels.RBF()(np.array([1., 2., 3., 4.]).reshape(-1, 1))
k1 = kernels.RBF(length_scale=1)(np.array([[-2],[ 2]]).reshape(-1, 1))
print(f"k1: {k1}")



gpr = GaussianProcessRegressor(
    kernels.RBF(),
    optimizer=None
)

x_t = np.array(np.linspace(-2, 2, 10)).reshape(-1, 1)
print(f"x_t={x_t}")

y_t = np.array([-np.sin(3. * x) - x ** 2 + 0.7 * x for x in x_t]).reshape(-1, 1)
print(f"y_t={y_t}")

train_indexes = [0,9]
x_train = x_t[train_indexes]
y_train = y_t[train_indexes]

print(f"x_train={x_train} y_train={y_train}")

gpr.fit(x_train, y_train)
print(f"gpr.alpha_0 ={gpr.alpha_}")

test_indexes = [2,7]
print(f"x_t[test_indexes]={x_t[test_indexes]}")
m, s =  gpr.predict(x_t[test_indexes], True, False)
print(f"gpr.alpha_={gpr.alpha_} m={m}, s={s}")

