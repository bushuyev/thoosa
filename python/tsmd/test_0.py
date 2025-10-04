# pip install tsmd matplotlib numpy

import numpy as np
import matplotlib.pyplot as plt
from tsmd.tools.synthetic_signal import SignalGenerator

def mask_to_intervals(mask: np.ndarray):
    mask = mask.astype(int)
    diff = np.diff(np.r_[0, mask, 0])
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    return list(zip(starts, ends))

# ---- ONE motif, with amplitude given as a range (low, high)
gen = SignalGenerator(
    n_motifs=1,
    motif_length=[200],            # single motif class, target length ~200
    motif_amplitude=(2.0, 2.0),    # <-- IMPORTANT: range, not scalar, to match your TSMD build
    motif_fundamental=3,
    motif_type="Sin",              # or "Cubic"
    noise_amplitude=0.4,
    length_fluctuation=0.2,
    amplitude_fluctuation=0.3,
    sparsity=0.6,
    sparsity_fluctuation=0.4,
    walk_amplitude=0.2,
    min_rep=3,
    max_rep=5,
)

np.random.seed(123)                # reproducible
signal, labels = gen.generate()    # signal: (N,), labels: (1, N)
motif_mask = labels[0]
intervals = mask_to_intervals(motif_mask)

# ---- Plot
fig, ax = plt.subplots(figsize=(12, 4))
x = np.arange(len(signal))
ax.plot(x, signal, linewidth=1, label="signal")

for j, (s, e) in enumerate(intervals):
    ax.axvspan(s, e, alpha=0.2, label="motif region" if j == 0 else None)

ax.set_title("TSMD synthetic signal with one motif (shaded)")
ax.set_xlabel("time index")
ax.set_ylabel("value")
ax.legend()
plt.tight_layout()
plt.show()

print("Motif intervals [start, end):", intervals)
