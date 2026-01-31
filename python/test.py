import  torch

num_of_gpus = torch.cuda.device_count()
print(f"num_of_gpus={num_of_gpus}")

is_available = torch.cuda.is_available()
print(f"is_available={is_available}")




print("Ok")