import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Dataset with variable-length, multi-channel sequences
# Each sample is Tensor[C, L_i]
# -----------------------------
class MySeqDataset(Dataset):
    def __init__(self, sequences, labels):
        """
        sequences: List[Tensor[C, L_i]]
        labels:    List[int]
        """
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# -----------------------------
# Collate: pad in time dimension only
# -----------------------------
def collate_batch(batch):
    """
    batch: list of (seq, label)
      seq:   [C, L_i]
      label: int
    Returns:
      x_padded: [B, C, L_max]
      lengths:  [B]  (original L_i per sample)
      labels:   [B]
    """
    seqs, labels = zip(*batch)  # tuples

    # lengths in time dimension
    lengths = torch.tensor([s.size(1) for s in seqs], dtype=torch.long)

    # pad over time:
    # pad_sequence works on [T, *], so transpose to [L_i, C]
    seqs_LC = [s.transpose(0, 1) for s in seqs]     # [L_i, C]
    padded_LC = pad_sequence(seqs_LC, batch_first=True)  # [B, L_max, C]

    # back to [B, C, L_max] for Conv1d
    x_padded = padded_LC.transpose(1, 2)  # [B, C, L_max]

    labels = torch.tensor(labels, dtype=torch.long)

    return x_padded, lengths, labels


# -----------------------------
# Full model: Conv1d motif bank + Transformer over motif activations
# -----------------------------
class MotifSequenceModel(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_motifs: int,
            motif_width: int,
            stride: int,
            d_model: int,
            nhead: int,
            num_layers: int,
            num_classes: int,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.motif_width = motif_width
        self.stride = stride

        # "Motif bank" – each conv filter is a motif detector over time
        self.motif_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_motifs,
            kernel_size=motif_width,
            stride=stride,
        )

        # Project motif activations to Transformer d_model
        self.input_proj = nn.Linear(num_motifs, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,  # [B, T, d_model]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths):
        """
        x:       [B, C, L_max]  (padded batch)
        lengths: [B]            (original sequence lengths, before padding)
        """
        # --- Motif Conv ---
        # conv_out: [B, num_motifs, T_max]
        conv_out = self.motif_conv(x)

        # Optional: turn into probabilities / bounded activations
        conv_out = torch.sigmoid(conv_out)

        # for Transformer: [B, T_max, num_motifs]
        conv_out = conv_out.transpose(1, 2)  # [B, T_max, F]
        B, T_max, F = conv_out.shape
        device = conv_out.device

        # --- Build mask in conv (motif) time domain ---
        # For each original length L_i, number of valid conv steps is:
        #   T_i = floor((L_i - kernel_size) / stride) + 1, if L_i >= kernel_size else 0
        valid_lengths = ((lengths - self.motif_width) // self.stride) + 1
        valid_lengths = torch.clamp(valid_lengths, min=0)       # avoid negative
        valid_lengths = torch.clamp(valid_lengths, max=T_max)   # avoid overflow

        # mask: True where time step is valid, False where it's padding
        time_ids = torch.arange(T_max, device=device).unsqueeze(0)  # [1, T_max]
        mask = time_ids < valid_lengths.unsqueeze(1)                # [B, T_max], bool

        # --- Transformer encoder ---
        x_enc_in = self.input_proj(conv_out)  # [B, T_max, d_model]

        # src_key_padding_mask: True where we want to IGNORE (padding)
        src_key_padding_mask = ~mask  # invert
        x_enc = self.encoder(x_enc_in, src_key_padding_mask=src_key_padding_mask)  # [B, T_max, d_model]

        # --- Masked mean pooling over time ---
        mask_f = mask.unsqueeze(-1).float()       # [B, T_max, 1]
        x_masked = x_enc * mask_f                 # zero out padded positions
        summed = x_masked.sum(dim=1)              # [B, d_model]
        denom = mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1] (avoid /0)
        pooled = summed / denom                   # [B, d_model]

        # --- Classification head ---
        logits = self.cls_head(pooled)            # [B, num_classes]
        return logits


# -----------------------------
# Tiny demo / training sketch
# -----------------------------
def main():
    # Hyperparams
    in_channels = 3           # your "3 channels" case
    motif_width = 20
    stride = 1
    num_motifs = 16
    d_model = 64
    nhead = 4
    num_layers = 2
    num_classes = 3

    # Fake data: 3 sequences, 3 channels, different lengths
    sequences = [
        torch.randn(in_channels, 150),
        torch.randn(in_channels, 220),
        torch.randn(in_channels, 87),
    ]
    labels = [0, 2, 1]

    dataset = MySeqDataset(sequences, labels)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,  # keep it simple; adjust as needed
    )

    model = MotifSequenceModel(
        in_channels=in_channels,
        num_motifs=num_motifs,
        motif_width=motif_width,
        stride=stride,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2):
        for x_padded, lengths, y in loader:
            # x_padded: [B, C, L_max]
            # lengths:  [B]
            # y:        [B]
            logits = model(x_padded, lengths)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"epoch {epoch}, loss = {loss.item():.4f}")


if __name__ == "__main__":
    main()
