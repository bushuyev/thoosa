import torch

def z_norm(x):
    var, mean = torch.var_mean(
        x,
        dim=(1, 2),
        correction=0,
        keepdim=True,
    )

    std = torch.sqrt(var + 1e-8)

    return (x - mean) / std


compiled = torch.compile(
    z_norm,
    fullgraph=True,
)


x = torch.rand(
    (100000, 2, 3),
    device="cuda",
)

for _ in range(10):
    y = compiled(x)

torch.cuda.synchronize()