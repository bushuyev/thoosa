import torch
#
#
# t = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7])
# print(t[3:5])
# torch_sum = torch.sum(t, dim=(0))
# print(torch_sum)

t_1 = torch.Tensor([[2, 4],[2, 4]])
print(t_1.size())

t_2 = torch.Tensor([[2],[2]])
print(t_2.size())

print(t_1/t_2)