#!/usr/bin/env python3
"""
Generate synthetic time series with planted motifs using TSMD's SignalGenerator
and export:
  - Data/series_{i}.csv                (index,value)
  - MetaData/series_{i}.json           (series_length + motif intervals)
  - motif_locations.csv                (series_file,motif_id,start,end)
  - Global_MetaData.json               (run config)

Motif intervals are derived from the binary label mask returned by TSMD:
labels.shape == (n_motifs, n_samples), 1 indicates motif active at that index.
"""

import argparse, json, os
from typing import List, Tuple
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

from tsmd.tools.synthetic_signal import SignalGenerator  # TSMD API


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def mask_to_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Convert a 1D binary mask (length N) into a list of [start, end) intervals
    where contiguous 1's form an interval.
    """
    assert mask.ndim == 1
    N = mask.shape[0]
    intervals = []
    i = 0
    while i < N:
        if mask[i] == 1:
            start = i
            i += 1
            while i < N and mask[i] == 1:
                i += 1
            intervals.append((start, i))
        else:
            i += 1
    return intervals


def write_series_csv(path: str, signal: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    # Friendly format for Rust CSV readers
    if pd is not None:
        pd.DataFrame({"index": np.arange(len(signal)), "value": signal.astype(float)}).to_csv(
            path, index=False
        )
    else:
        with open(path, "w") as f:
            f.write("index,value\n")
            for i, v in enumerate(signal):
                f.write(f"{i},{float(v)}\n")


def write_meta_json(path: str, series_file: str, series_len: int, motif_intervals: List[List[Tuple[int, int]]]) -> None:
    ensure_dir(os.path.dirname(path))
    payload = {
        "source_id": series_file,
        "series_length": int(series_len),
        "motifs": [
            {
                "motif_id": motif_id,
                "instances": [{"start": s, "end": e} for (s, e) in intervals],
            }
            for motif_id, intervals in enumerate(motif_intervals)
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_locations_csv(path: str, rows: List[Tuple[str, int, int, int]]) -> None:
    ensure_dir(os.path.dirname(path))
    if pd is not None:
        pd.DataFrame(rows, columns=["series_file", "motif_id", "start", "end"]).to_csv(path, index=False)
    else:
        with open(path, "w") as f:
            f.write("series_file,motif_id,start,end\n")
            for sf, mid, s, e in rows:
                f.write(f"{sf},{mid},{s},{e}\n")


def main():
    p = argparse.ArgumentParser(description="Generate TSMD synthetic series with ground-truth motif locations.")
    p.add_argument("--out", required=True, help="Output dir (creates Data/ and MetaData/).")
    p.add_argument("--num-series", type=int, default=5)
    p.add_argument("--n-motifs", type=int, default=3, help="Number of different motif types in each series.")
    p.add_argument("--motif-length", type=int, nargs="+", default=[200],
                   help="Either one length (applied to all motifs) or exactly n_motifs lengths.")
    p.add_argument("--motif-amplitude", type=float, default=2.0)
    p.add_argument("--motif-fundamental", type=int, default=3, help="Complexity of base motifs.")
    p.add_argument("--motif-type", choices=["Sin", "Cubic"], default="Sin")
    p.add_argument("--noise-amplitude", type=float, default=0.5)
    p.add_argument("--length-fluctuation", type=float, default=0.3)
    p.add_argument("--amplitude-fluctuation", type=float, default=0.3)
    p.add_argument("--sparsity", type=float, default=0.5, help="Gap = motif_length * sparsity.")
    p.add_argument("--sparsity-fluctuation", type=float, default=0.5)
    p.add_argument("--walk-amplitude", type=float, default=0.2, help="Random-walk background amplitude.")
    p.add_argument("--min-rep", type=int, default=5)
    p.add_argument("--max-rep", type=int, default=10)
    p.add_argument("--exact-occur", type=int, nargs="*", default=None,
                   help="Exact repetitions per motif (length n_motifs); overrides min/max if set.")
    p.add_argument("--exact-ts-length", type=int, default=None,
                   help="Force exact total length; generator pads gaps accordingly.")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    out_root = os.path.abspath(args.out)
    data_dir = os.path.join(out_root, "Data")
    meta_dir = os.path.join(out_root, "MetaData")
    ensure_dir(data_dir); ensure_dir(meta_dir)

    # Validate motif_length vector vs n_motifs
    motif_length = args.motif_length
    if len(motif_length) == 1:
        motif_length = motif_length * args.n_motifs
    elif len(motif_length) != args.n_motifs:
        raise SystemExit(f"--motif-length must have length 1 or n_motifs ({args.n_motifs}). Got {len(motif_length)}.")

    # Validate exact occurrences, if given
    exact_occur = None
    if args.exact_occur:
        if len(args.exact_occur) != args.n_motifs:
            raise SystemExit(f"--exact-occur must have length n_motifs ({args.n_motifs}). Got {len(args.exact_occur)}.")
        exact_occur = args.exact_occur

    rng = np.random.default_rng(args.seed)
    all_rows: List[Tuple[str, int, int, int]] = []

    # Save run config
    with open(os.path.join(out_root, "Global_MetaData.json"), "w") as f:
        json.dump({
            "num_series": args.num_series,
            "n_motifs": args.n_motifs,
            "motif_length": motif_length,
            "motif_amplitude": args.motif_amplitude,
            "motif_fundamental": args.motif_fundamental,
            "motif_type": args.motif_type,
            "noise_amplitude": args.noise_amplitude,
            "length_fluctuation": args.length_fluctuation,
            "amplitude_fluctuation": args.amplitude_fluctuation,
            "sparsity": args.sparsity,
            "sparsity_fluctuation": args.sparsity_fluctuation,
            "walk_amplitude": args.walk_amplitude,
            "min_rep": args.min_rep,
            "max_rep": args.max_rep,
            "exact_occurences": exact_occur,
            "exact_ts_length": args.exact_ts_length,
            "seed": args.seed,
            "data_dir": "Data",
            "meta_dir": "MetaData",
            "locations_file": "motif_locations.csv"
        }, f, indent=2)

    for i in range(args.num_series):
        # Different seed per series for reproducibility but diversity
        local_seed = int(rng.integers(0, 2**31 - 1))

        gen = SignalGenerator(
            n_motifs=args.n_motifs,
            motif_length=motif_length,
            motif_amplitude=args.motif_amplitude,
            motif_fundamental=args.motif_fundamental,
            motif_type=args.motif_type,
            noise_amplitude=args.noise_amplitude,
            length_fluctuation=args.length_fluctuation,
            amplitude_fluctuation=args.amplitude_fluctuation,
            sparsity=args.sparsity,
            sparsity_fluctuation=args.sparsity_fluctuation,
            walk_amplitude=args.walk_amplitude,
            min_rep=args.min_rep,
            max_rep=args.max_rep,
            exact_occurences=exact_occur,
            exact_ts_length=args.exact_ts_length,
        )

        # TSMD's generator is pseudo-random via numpy's global RNG; set seed:
        np.random.seed(local_seed)

        signal, labels = gen.generate()  # signal: (N,), labels: (n_motifs, N)

        # Export series
        series_file = f"series_{i}.csv"
        write_series_csv(os.path.join(data_dir, series_file), signal)

        # Convert each motif’s binary row into intervals
        motif_intervals: List[List[Tuple[int, int]]] = []
        for motif_id in range(labels.shape[0]):
            intervals = mask_to_intervals(labels[motif_id].astype(int))
            motif_intervals.append(intervals)
            # rows for global csv
            for (s, e) in intervals:
                all_rows.append((series_file, motif_id, int(s), int(e)))

        # Per-series JSON metadata
        write_meta_json(
            os.path.join(meta_dir, f"series_{i}.json"),
            series_file=series_file,
            series_len=signal.shape[0],
            motif_intervals=motif_intervals,
        )

    # Global locations CSV
    write_locations_csv(os.path.join(out_root, "motif_locations.csv"), all_rows)

    print(f"[OK] Wrote {args.num_series} series into:\n"
          f"  {data_dir}/series_*.csv\n"
          f"  {meta_dir}/series_*.json\n"
          f"  {out_root}/motif_locations.csv\n"
          f"  {out_root}/Global_MetaData.json")
    

if __name__ == "__main__":
    main()

