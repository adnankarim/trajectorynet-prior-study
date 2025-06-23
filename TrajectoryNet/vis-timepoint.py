# vis.py

import argparse
import os

import matplotlib.pyplot as plt
from TrajectoryNet.dataset import SCData  # <-- make sure dataset_repro.py (with your SCData+factory) is in the same folder

def main():
    p = argparse.ArgumentParser(
        description="Plot one timepoint of a TrajectoryNet SCData dataset"
    )
    p.add_argument(
        "--dataset",
        default="CIRCLE5",
        help="Dataset name (e.g. CYCLE, CIRCLE3, TREE, MOONS, etc.)",
    )
    p.add_argument(
        "--timepoint",
        type=float,
        default=0,
        help="Which timepoint to plot (must exactly match one of data.get_times()).",
    )
    p.add_argument(
        "--out",
        default="img/cycle_t0.png",
        help="Where to write the PNG (e.g. img/cycle_t0.png).",
    )
    args = p.parse_args()

    name = args.dataset.upper()
    data_obj = SCData.factory(name, args)

    data = data_obj.get_data()
    times = data_obj.get_times()

    mask = times == args.timepoint
    if not mask.any():
        raise ValueError(
            f"No points at timepoint {args.timepoint} in dataset {name}"
        )
    pts = data[mask]

    if pts.shape[1] < 2:
        raise ValueError("Need at least 2 dimensions to scatter‐plot.")
    if pts.shape[1] > 2:
        print(
            f"Warning: data has {pts.shape[1]} dims; plotting dims 0 & 1 only."
        )

    plt.figure(figsize=(6, 6))
    plt.scatter(pts[:, 0], pts[:, 1], alpha=0.7, edgecolors="none")
    plt.title(f"{name} @ t={args.timepoint}")
    plt.xlabel("dim 0")
    plt.ylabel("dim 1")
    plt.tight_layout()

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=300)
    print(f"Saved plot to {args.out}")

if __name__ == "__main__":
    main()
