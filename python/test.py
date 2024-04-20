import torch
import numpy as np
#
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

k = np.array([
    [
        [1, 11], [2, 22], [3, 33]
    ],
    [
        [10, 1010], [20, 2020], [30, 3030]
    ],
    [
        [100, 100100], [200, 200200], [300, 300300]
    ],
    [
        [1000, 10001000], [2000, 20002000], [3000, 30003000]
    ],
]
)

print(f"shape={k.shape} k={k}")
print(f"k.T.shape={k.T.shape} k.T={k.T}")


t_1 = torch.Tensor(k)

print(f"k.T.shape={t_1.T.shape} k.T={t_1.T}")



