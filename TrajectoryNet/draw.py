import argparse
import os
import matplotlib.pyplot as plt
from TrajectoryNet.dataset import SCData

def plot_dataset(ax, dataset_name, timepoints, plot_label=None):
    data_obj = SCData.factory(dataset_name, argparse.Namespace())
    data = data_obj.get_data()
    times = data_obj.get_times()
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', '<', '>']
    for i, tp in enumerate(timepoints):
        mask = times == tp
        if not mask.any():
            print(f"Warning: No points at timepoint {tp} in {dataset_name}")
            continue
        pts = data[mask]
        if pts.shape[1] < 2:
            raise ValueError(f"{dataset_name} needs at least 2 dimensions to plot.")
        ax.scatter(
            pts[:, 0], pts[:, 1],
            alpha=0.7,
            edgecolors="none",
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=f"t={i}"
        )
    ax.set_title(f"{plot_label or dataset_name}")
    ax.set_xlabel("dim 0")
    ax.set_ylabel("dim 1")
    ax.legend()

def main():
    p = argparse.ArgumentParser(description="Plot multiple timepoints for TREE, ARCH, and CYCLE datasets")
    p.add_argument("--timepoints", type=float, nargs='+', required=True, help="List of timepoints to plot (space-separated, e.g., --timepoints 1 2)")
    p.add_argument("--out", default="img/grid.png", help="Optional: where to write PNG (e.g. img/grid.png).")
    args = p.parse_args()

    datasets = [("TREE", "TREE"), ("CIRCLE5", "ARCH"), ("CYCLE", "CYCLE")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, (data_name, plot_label) in enumerate(datasets):
        plot_dataset(axs[i], data_name, args.timepoints, plot_label=plot_label)
    plt.tight_layout()

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(args.out, dpi=300)
        print(f"Saved grid plot to {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()



# >python -m TrajectoryNet.draw --timepoints 1 2