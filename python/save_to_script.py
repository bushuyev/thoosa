import torch


class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(1, 1),
            torch.nn.Linear(4, 4)
        )

    def forward(self, x, h):
        new_h = torch.tanh(self.seq(x) + h)
        return new_h, new_h

my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell =  torch.jit.script(my_cell)
print(traced_cell.code)

# traced_cell(x, h)