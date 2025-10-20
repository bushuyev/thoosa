import torch
import math
import pytest


def motif_specificity(S, N, alpha=0.5, lam=10.0):
    # S: [M, L] (series-with-motif per label), N: [L] (series per label)
    S = S.to(torch.float64)
    N = N.to(torch.float64)
    M, L = S.shape

    # Smoothing
    S_s = S + alpha
    N_s = N + L*alpha

    S_m = S_s.sum(dim=1, keepdim=True)               # [M,1]
    q = S_s / S_m                                    # q_{mℓ} = P(label=ℓ | motif m)
    purity = (q.pow(2).sum(dim=1) - 1.0/L) / (1.0 - 1.0/L)  # [M]

    # P(m | ℓ) and P(m)
    P_m_given_l = S_s / N_s                          # [M,L]
    P_m = (S_s.sum(dim=1) / N_s.sum()).unsqueeze(1)  # [M,1]

    lift = P_m_given_l / P_m                         # [M,L]
    lift_top, top_idx = lift.max(dim=1)              # [M], [M]

    support = S_m.squeeze(1) - L*alpha               # back to unsmoothed approx support
    w = support / (support + lam)

    spec = purity * torch.clamp(lift_top.log2(), min=0.0) * w
    return spec, top_idx, purity, lift_top, support



def test_perfect_specificity_two_labels_exact_value():
    """
    Two labels, two motifs. Motif 0 appears only in label 0; motif 1 is uniform.
    With alpha=0, lam=10 we can compute an exact expected spec for motif 0.
    """
    # S: [M, L] = series-level presence counts per label
    S = torch.tensor([
        [10, 0],  # motif 0: perfectly specific to label 0
        [5,  5],  # motif 1: uniform across labels
    ])
    N = torch.tensor([50, 50])  # 50 series in each label

    spec, top_idx, purity, lift_top, support = motif_specificity(S, N, alpha=0.0, lam=10.0)

    # Motif 0 expectations:
    # purity = 1, lift_top = 2, log2(lift_top) = 1, support=10, w = 10/(10+10)=0.5
    # spec = 1 * 1 * 0.5 = 0.5
    assert top_idx[0].item() == 0
    assert pytest.approx(purity[0].item(), rel=1e-6) == 1.0
    assert pytest.approx(lift_top[0].item(), rel=1e-6) == 2.0
    assert pytest.approx(spec[0].item(), rel=1e-6) == 0.5

    # Motif 1: uniform → purity=0, lift=1 → log2(lift)=0 → spec=0
    assert pytest.approx(purity[1].item(), rel=1e-6) == 0.0
    assert pytest.approx(lift_top[1].item(), rel=1e-6) == 1.0
    assert pytest.approx(spec[1].item(), rel=1e-6) == 0.0


def test_imbalance_lift_and_support_monotonicity():
    """
    Strong class imbalance (900 vs 100). The motif is proportionally enriched in the minority class.
    Check that top label is the minority class, lift > 1, and scaling support up increases spec.
    """
    N = torch.tensor([900, 100])

    # Motif A: low support but enriched in label 1 (minority)
    # P(m|1)=20/100 = 0.2, P(m)=29/1000 = 0.029 => lift ≈ 6.9, log2 ≈ 2.79
    S = torch.tensor([
        [ 9,  20],   # motif A
        [90, 200],   # motif B = 10× motif A (same proportions, higher support)
    ])

    spec, top_idx, purity, lift_top, support = motif_specificity(S, N, alpha=0.0, lam=10.0)

    # Both motifs should point to label 1 (minority)
    assert top_idx.tolist() == [1, 1]

    # Enrichment is substantial
    assert lift_top[0].item() > 4.0
    assert lift_top[1].item() > 4.0

    # Same proportions -> same purity and lift; higher support should increase spec via w
    assert purity[1].item() == pytest.approx(purity[0].item(), rel=1e-12)
    assert lift_top[1].item() == pytest.approx(lift_top[0].item(), rel=1e-12)
    assert spec[1].item() > spec[0].item()


def test_all_zero_counts_yield_zero_specificity_with_smoothing():
    """
    If a motif never appears (all zeros), with alpha>0:
    q is uniform -> purity=0; lift_top=1 -> log2=0; so spec=0.
    """
    S = torch.tensor([[0, 0, 0]])  # one motif, three labels
    N = torch.tensor([30, 40, 50])

    spec, top_idx, purity, lift_top, support = motif_specificity(S, N, alpha=0.5, lam=10.0)

    assert pytest.approx(purity[0].item(), rel=1e-9) == 0.0
    assert pytest.approx(lift_top[0].item(), rel=1e-9) == 1.0
    assert pytest.approx(spec[0].item(), rel=1e-9) == 0.0