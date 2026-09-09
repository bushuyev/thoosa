#!/usr/bin/env python3

import argparse
import math
import sys
from typing import List

import torch


def parse_size(s: str) -> int:
    s = s.strip().lower()

    units = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "k": 1024,
        "m": 1024**2,
        "g": 1024**3,
    }

    for suffix, multiplier in sorted(units.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * multiplier)

    return int(s)


def human_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n} B"


def default_sizes() -> List[int]:
    return [
        4 * 1024,
        16 * 1024,
        64 * 1024,
        256 * 1024,
        1 * 1024**2,
        4 * 1024**2,
        16 * 1024**2,
        64 * 1024**2,
        256 * 1024**2,
        512 * 1024**2,
        1024 * 1024**2,
        ]


def choose_iters(nbytes: int, target_total_bytes: int, min_iters: int, max_iters: int) -> int:
    iters = math.ceil(target_total_bytes / nbytes)
    iters = max(min_iters, iters)
    iters = min(max_iters, iters)
    return iters


def bandwidth_gbps(nbytes: int, seconds: float, iters: int) -> float:
    total_bits = nbytes * 8 * iters
    return total_bits / seconds / 1e9


def bandwidth_gbs(nbytes: int, seconds: float, iters: int) -> float:
    total_bytes = nbytes * iters
    return total_bytes / seconds / 1e9


def bench_h2d(cpu_src: torch.Tensor, gpu_dst: torch.Tensor, stream: torch.cuda.Stream, iters: int, warmup: int) -> float:
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            gpu_dst.copy_(cpu_src, non_blocking=True)

    stream.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(iters):
            gpu_dst.copy_(cpu_src, non_blocking=True)
        end.record(stream)

    end.synchronize()
    return start.elapsed_time(end) / 1000.0


def bench_d2h(gpu_src: torch.Tensor, cpu_dst: torch.Tensor, stream: torch.cuda.Stream, iters: int, warmup: int) -> float:
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            cpu_dst.copy_(gpu_src, non_blocking=True)

    stream.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(iters):
            cpu_dst.copy_(gpu_src, non_blocking=True)
        end.record(stream)

    end.synchronize()
    return start.elapsed_time(end) / 1000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="Comma-separated sizes, e.g. 4KiB,64KiB,1MiB,64MiB,1GiB",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--min-iters", type=int, default=5)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument(
        "--target-total",
        type=str,
        default="8GiB",
        help="Approximate total bytes copied per direction per size",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available in this PyTorch build.", file=sys.stderr)
        return 1

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    stream = torch.cuda.Stream(device=device)

    if args.sizes:
        sizes = [parse_size(x) for x in args.sizes.split(",")]
    else:
        sizes = default_sizes()

    target_total_bytes = parse_size(args.target_total)

    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA:         {torch.version.cuda}")
    print(f"Device:       cuda:{args.device} - {torch.cuda.get_device_name(args.device)}")
    print(f"Pinned host:  yes")
    print(f"Dtype:        uint8, so tensor.numel() == transfer bytes")
    print()

    header = (
        f"{'Size':>12}  "
        f"{'Iters':>8}  "
        f"{'H2D GB/s':>10}  "
        f"{'H2D Gb/s':>10}  "
        f"{'D2H GB/s':>10}  "
        f"{'D2H Gb/s':>10}"
    )
    print(header)
    print("-" * len(header))

    for nbytes in sizes:
        iters = choose_iters(
            nbytes=nbytes,
            target_total_bytes=target_total_bytes,
            min_iters=args.min_iters,
            max_iters=args.max_iters,
        )

        # Efficient host buffers: page-locked / pinned memory.
        cpu_src = torch.empty((nbytes,), dtype=torch.uint8, pin_memory=True)
        cpu_dst = torch.empty((nbytes,), dtype=torch.uint8, pin_memory=True)

        # Preallocated device buffer.
        gpu_buf = torch.empty((nbytes,), dtype=torch.uint8, device=device)

        # Touch memory once outside benchmark.
        cpu_src.fill_(123)
        cpu_dst.fill_(0)
        gpu_buf.fill_(0)
        torch.cuda.synchronize(device)

        h2d_sec = bench_h2d(
            cpu_src=cpu_src,
            gpu_dst=gpu_buf,
            stream=stream,
            iters=iters,
            warmup=args.warmup,
        )

        d2h_sec = bench_d2h(
            gpu_src=gpu_buf,
            cpu_dst=cpu_dst,
            stream=stream,
            iters=iters,
            warmup=args.warmup,
        )

        h2d_gbs = bandwidth_gbs(nbytes, h2d_sec, iters)
        h2d_gbps = bandwidth_gbps(nbytes, h2d_sec, iters)

        d2h_gbs = bandwidth_gbs(nbytes, d2h_sec, iters)
        d2h_gbps = bandwidth_gbps(nbytes, d2h_sec, iters)

        print(
            f"{human_bytes(nbytes):>12}  "
            f"{iters:8d}  "
            f"{h2d_gbs:10.2f}  "
            f"{h2d_gbps:10.2f}  "
            f"{d2h_gbs:10.2f}  "
            f"{d2h_gbps:10.2f}"
        )

        del cpu_src, cpu_dst, gpu_buf
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())