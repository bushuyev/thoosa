import torch
import torch.nn as nn

class LSTMCellFromScratch(nn.Module):
    """
    One LSTM cell step (no nn.LSTM / no nn.LSTMCell).
    Gate order: i, f, g, o  (same as PyTorch).
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size


        # Combine all gates into one big affine for x and h:
        # gates = x @ W_ih^T + b_ih + h @ W_hh^T + b_hh
        self.W_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size) * 0.1)
        self.W_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size) * 0.1)
        self.b_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(self, x_t, state):
        """
        x_t:   (B, input_size)
        state: (h_{t-1}, c_{t-1}) each (B, hidden_size)
        returns: (h_t, c_t)
        """
        h_prev, c_prev = state

        gates = (x_t @ self.W_ih.T + self.b_ih) + (h_prev @ self.W_hh.T + self.b_hh)
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)      # input gate
        f = torch.sigmoid(f)      # forget gate
        g = torch.tanh(g)         # candidate
        o = torch.sigmoid(o)      # output gate

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


class LSTMFromScratch(nn.Module):
    """Runs the above cell over a full sequence."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = LSTMCellFromScratch(input_size, hidden_size)

    def forward(self, x, h0=None, c0=None):
        """
        x: (B, T, input_size)
        returns:
          y:  (B, T, hidden_size)  all hidden states
          hT: (B, hidden_size)     final h
          cT: (B, hidden_size)     final c
        """
        B, T, _ = x.shape
        H = self.cell.hidden_size

        if h0 is None: h0 = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        if c0 is None: c0 = torch.zeros(B, H, device=x.device, dtype=x.dtype)

        h, c = h0, c0
        ys = []
        for t in range(T):
            h, c = self.cell(x[:, t, :], (h, c))
            ys.append(h)

        y = torch.stack(ys, dim=1)  # (B, T, H)
        return y, h, c


# --- tiny demo ---
B, T, D_in, H = 2, 5, 3, 4
x = torch.randn(B, T, D_in)

model = LSTMFromScratch(D_in, H)
y, hT, cT = model(x)

print("x :", x.shape)   # (B, T, D_in)
print("y :", y.shape)   # (B, T, H)
print("hT:", hT.shape)  # (B, H)
print("cT:", cT.shape)  # (B, H)