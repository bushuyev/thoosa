import torch
from torch.distributions.normal import Normal
from matplotlib import pyplot
import numpy as np
from scipy.stats import norm, entropy
from scipy.special import erf

# dist = Normal(torch.tensor([0]), torch.tensor([3]))
# print(dist.cdf(torch.tensor([3])))


# dist = Normal(torch.tensor([0,0]), torch.tensor([3,3]))
# print(dist.cdf(torch.tensor([0,0])))


# pyplot.hist(dist.sample(torch.Size((1000,))))
# pyplot.show()
mu_x = torch.tensor([0,0])
sigma_x = torch.tensor([3,3])

incumbent = torch.tensor([0])
xi = 0.01

imp = mu_x - incumbent - xi
Z = imp / sigma_x
ei = imp * norm.cdf(Z) + sigma_x * norm.pdf(Z)

print(f"ei={ei} cdf={norm.cdf(Z)} norm.pdf(Z)={norm.pdf(Z)}")

# gamma_x = (mu_x - (incumbent + kappa)) / sigma_x
# print(f"gamma_x={gamma_x}")
# # Compute EI based on some temporary variables that can be reused in
# # gradient computation
# tmp_erf = erf(gamma_x / np.sqrt(2))
# tmp_ei_no_std = 0.5*gamma_x * (1 + tmp_erf) + np.exp(-gamma_x**2/2)/np.sqrt(2*np.pi)
#
# print(f"tmp_ei_no_std={tmp_ei_no_std}")
#
# ei = sigma_x * tmp_ei_no_std
#
# print(f"ei={ei}")