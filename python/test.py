import torch


t = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7])
print(t[3:5])
torch_sum = torch.sum(t, dim=(0))
print(torch_sum)