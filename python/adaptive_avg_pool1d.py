import  torch

num_of_gpus = torch.cuda.device_count()
print(f"num_of_gpus={num_of_gpus}")

is_available = torch.cuda.is_available()
print(f"is_available={is_available}")



N, C, L = 1000, 100, 100
num_features = 100

tensors = [torch.randn(N, C, L, device="cuda", requires_grad=True) for i in range(100)]

x = torch.concat(tensors, 1)

print(f"x.size={x.size()}")

y = torch.nn.functional.adaptive_avg_pool1d(x.flatten(1,2), num_features)
loss = y.sum()
loss.backward()


print("Ok")