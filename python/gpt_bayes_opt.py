import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import norm
from scipy.optimize import minimize
from matplotlib import pyplot

class GaussianProcessRegressor:
    def __init__(self, kernel, alpha=1e-5):
        self.kernel = kernel
        self.alpha = alpha

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        K = self.kernel(self.X_train, self.X_train)
        self.L = cholesky(K + self.alpha * np.eye(len(self.X_train)), lower=True)
        self.alpha_ = solve_triangular(self.L.T, solve_triangular(self.L, self.y_train, lower=True))

    def predict(self, X):
        K_s = self.kernel(self.X_train, X)
        K_ss = self.kernel(X, X)
        Lk = solve_triangular(self.L, K_s, lower=True)
        mu = np.dot(Lk.T, self.alpha_)
        v = solve_triangular(self.L, Lk, lower=True)
        sigma = np.sqrt(np.diag(K_ss) - np.sum(v**2, axis=0))
        return mu, sigma

class BayesianOptimization:
    def __init__(self, objective_function, bounds, initial_points=3):
        self.objective_function = objective_function
        self.bounds = bounds
        self.dim = len(bounds)
        self.initial_points = initial_points
        self.X = np.random.uniform(bounds[:, 0], bounds[:, 1], (initial_points, self.dim))
        self.Y = np.array([self.objective_function(x) for x in self.X])
        self.min_y = np.min(self.Y)

    def expected_improvement(self, x, gp, y_min, xi=0.01):
        mu, sigma = gp.predict(x.reshape(1, -1))
        sigma = np.maximum(sigma, 1e-9)
        z = (mu - y_min - xi) / sigma
        return (mu - y_min - xi) * norm.cdf(z) + sigma * norm.pdf(z)

    def acquisition_function(self, x, gp, y_min):
        return -self.expected_improvement(x, gp, y_min)

    def fit_gp(self, X, Y):
        kernel = lambda x, y: np.exp(-0.5 * np.subtract.outer(x, y)**2)
        gp = GaussianProcessRegressor(kernel)
        gp.fit(X, Y)
        return gp

    def find_next_point(self, gp):
        res = minimize(lambda x: self.acquisition_function(x, gp, self.min_y),
                       x0=np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim),
                       bounds=self.bounds)
        return res.x

    def optimize(self, n_iter=10):
        for _ in range(n_iter):
            gp = self.fit_gp(self.X, self.Y)
            x_next = self.find_next_point(gp)
            y_next = self.objective_function(x_next)
            self.X = np.vstack((self.X, x_next))
            self.Y = np.append(self.Y, y_next)
            self.min_y = min(self.min_y, y_next)

# Example usage:
def objective_function(x):
    return -np.sin(3*x) - x**2 + 0.7*x

bounds = np.array([[-2, 2]])

bayesian_opt = BayesianOptimization(objective_function, bounds)
# bayesian_opt.optimize(n_iter=10)


x_1 = np.asarray(np.arange(0, 1, 0.01))
pyplot.plot(x_1, [objective_function(x) for x in x_1], 'g')
pyplot.show()


