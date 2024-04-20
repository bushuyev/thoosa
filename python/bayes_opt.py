# example of a gaussian process surrogate function
from math import sin
from math import pi

import numpy as np
from numpy import arange
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from matplotlib import pyplot
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor

# objective function
def objective(x, noise=0.1):
    noise = normal(loc=0, scale=noise)
    return (x**2 * sin(5 * pi * x)**6.0) + noise

# surrogate or approximation for the objective function
def surrogate(model, X):
    # catch any warning generated when making a prediction
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
        return model.predict(X, return_std=True)

# plot real observations vs surrogate function
def plot(X, y, model):
    # scatter plot of inputs and real objective function
    pyplot.scatter(X, y)
    # line plot of surrogate function across domain
    Xsamples = asarray(arange(0, 1, 0.001))
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    ysamples, _ = surrogate(model, Xsamples)
    pyplot.plot(Xsamples, ysamples)
    # show the plot
    pyplot.show()

# sample the domain sparsely with noise
X = asarray([0.1, 0.2, 0.3]) # random(10) #asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) #random(10)
y_obj = asarray([objective(x, noise=0.001) for x in X])
# reshape into rows and cols
X = X.reshape(len(X), 1)
y_obj = y_obj.reshape(len(y_obj), 1)

model = GaussianProcessRegressor()
model.fit(X, y_obj)


x_1 = asarray(arange(0, 1, 0.01))

pred_y = model.predict(x_1.reshape(len(x_1), 1),return_std=False)
print(f"pred_y={pred_y.shape}")

pyplot.scatter(X, y_obj)
pyplot.plot(x_1, [objective(x, noise=0) for x in x_1], 'g')
pyplot.plot(x_1, pred_y, 'r')
# pyplot.plot(x_1, model.predict())
pyplot.show()