import torch
from torch import optim
from matplotlib import pyplot

# Define the parabola function
def func(x:torch.Tensor):
    return (x - 20) ** 2 + (x).sin()
    # x_0 = (x-50)/10
    # return  x_0 ** 2 + 100 * x_0.sin()

# Initialize the variable to optimize (parameter)
x = torch.tensor([0.0], requires_grad=True)

# Define the LBFGS optimizer
optimizer = optim.LBFGS([x])#, lr=0.01, max_iter=1000, line_search_fn='strong_wolfe')

# Closure to evaluate the function and its gradient
def closure():
    optimizer.zero_grad()  # Zero the gradients
    output = func(x)   # Evaluate the function
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

pyplot.plot(range(-100,100), [func(torch.tensor(x)) for x in range(-100, 100)], 'r')
pyplot.show()
