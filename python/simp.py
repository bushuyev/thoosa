from shared_util import plot_train_stats, clc_cnn_l_out, train, clc_cnn_max_pool_out, TrainCfg
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch

sin_y = np.array([4. * np.sin(np.pi * x / 20.) for x in range(0, 100)])
x2_y = np.array([2. * (x / 50.) ** 2 - 4. for x in range(0, 100)])


class NPDataset(Dataset):

    def __getitem__(self, idx):
        if idx % 2 == 0:
            return torch.tensor(sin_y, dtype=torch.float), torch.tensor(0, dtype=torch.float)
        else:
            return torch.tensor(x2_y, dtype=torch.float), torch.tensor(1, dtype=torch.float)

    def __len__(self):
        return 10



plt.plot(sin_y, 'r', label="sin")
plt.plot(x2_y, 'g', label="x2")

plt.show()


class Simp(nn.Module):

    def __init__(self):
        super(Simp, self).__init__()

        self.conv_1 = nn.Conv1d(1, 1, kernel_size=10, stride=1, padding = 3)
        self.pool = nn.MaxPool1d(kernel_size=10, stride=1, padding=3)
        lin_feats = clc_cnn_max_pool_out(
            clc_cnn_l_out(100, dilation=1, kernel_size=10, padding=3, stride=1),
            1, 10, 3, 1
        )
        self.lin = nn.Linear(lin_feats, 1)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.conv_1.weight.data)
        self.lin.bias.data.zero_()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.pool(x)
        x = self.lin(x)
        x = self.sigmoid(x)
        return x

    def configure_optimizers(self, cfg):
        return torch.optim.AdamW(self.parameters())


model = Simp()
plot_train_stats(
    train(
        model,
        NPDataset(),
        y_view=lambda y: y.view(1,1),
        loss_fn=nn.BCELoss(),
        cfg=TrainCfg(cycles=30)
    )
)

print("")

with torch.no_grad():
    model.eval()

    sin_1_t = torch.tensor(sin_y, dtype=torch.float)
    x2_y_t = torch.tensor(x2_y, dtype=torch.float)

    for _ in range(0,10):
        print(f"sin={model(sin_1_t.view(1, -1))}")
        print(f"x2={model(x2_y_t.view(1, -1))}")
        print("--------------------------------------")


print("Ok")