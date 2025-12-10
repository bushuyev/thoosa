import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Model definition (same as before)
class PatternPresence1D(nn.Module):
    def __init__(self, in_channels=1, num_filters=32, kernel_size=25):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)  # global max over time
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        # x: [B, C, T]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)          # [B, num_filters, 1]
        x = x.squeeze(-1)         # [B, num_filters]
        x = torch.sigmoid(self.fc(x))  # [B, 1] probability of pattern presence
        return x


# 2. Synthetic data: noise + pattern
def create_synthetic_batch(batch_size, seq_len, pattern, device):
    """
    Half of the batch: noise only (label 0)
    Half of the batch: noise + injected pattern at random position (label 1)
    """
    pattern = pattern.to(device)
    pat_len = pattern.shape[-1]

    x = 0.1 * torch.randn(batch_size, 1, seq_len, device=device)  # [B, 1, T]
    y = torch.zeros(batch_size, 1, device=device)                 # [B, 1]

    half = batch_size // 2
    for i in range(half):
        start = torch.randint(0, seq_len - pat_len, (1,), device=device).item()
        x[i, 0, start:start + pat_len] += pattern
        y[i, 0] = 1.0

    return x, y


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Settings
    seq_len = 200
    pattern_len = 40
    batch_size = 64
    num_steps = 500

    # Define a simple sinusoidal pattern
    t = torch.linspace(0, 2 * math.pi, steps=pattern_len)
    pattern = torch.sin(t)  # shape [pattern_len]

    # 3. Create model + optimizer + loss
    model = PatternPresence1D(in_channels=1,
                              num_filters=32,
                              kernel_size=pattern_len).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # 4. Training loop
    model.train()
    for step in range(1, num_steps + 1):
        x_batch, y_batch = create_synthetic_batch(batch_size, seq_len, pattern, device)
        preds = model(x_batch)               # [B, 1]
        loss = criterion(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            with torch.no_grad():
                pred_labels = (preds > 0.5).float()
                acc = (pred_labels == y_batch).float().mean().item()
            print(f"Step {step:4d} | loss={loss.item():.4f} | acc={acc:.3f}")

    # 5. Inference demo on a few new signals
    model.eval()
    with torch.no_grad():
        # Build a small test batch
        x_test, y_test = create_synthetic_batch(batch_size=4,
                                                seq_len=seq_len,
                                                pattern=pattern,
                                                device=device)
        probs = model(x_test)  # [4, 1]

    print("\nTest samples:")
    for i in range(4):
        true_label = int(y_test[i].item())
        prob = probs[i].item()
        print(f"  sample {i}: true={true_label}, predicted_prob={prob:.3f}")


if __name__ == "__main__":
    main()
