#!/usr/bin/env python3
"""
Inspect the synthetic TreeTestData dataset.

Usage
-----
$ python inspect_tree.py
"""
import matplotlib.pyplot as plt
import numpy as np

# ---- import the class -------------------------------------------------
from TrajectoryNet.dataset import TreeTestData   # adjust path if needed

def main():
    ds = TreeTestData()           # builds the 10 000-point dataset
    X      = ds.get_data()        # (N,2)
    labels = ds.get_times()       # (N,)

    # --- basic info ----------------------------------------------------
    print("data shape        :", X.shape)
    print("labels unique     :", np.unique(labels, return_counts=True))
    print("min/max x,y       :", X.min(0), X.max(0))
    print("velocity present? :", ds.has_velocity())

    # --- scatter plot coloured by time-point ---------------------------
    plt.figure(figsize=(5,5))
    cmap = {0: "tab:blue", 1: "tab:orange"}
    for tp in np.unique(labels):
        m = labels == tp
        plt.scatter(X[m,0], X[m,1], s=4, alpha=.6, c=cmap[tp], label=f"tp={tp}")
    plt.axis("equal")
    plt.title("TreeTestData – 2 D embedding")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tree_dataset.png", dpi=300)  # Save image
    print("Saved plot to tree_dataset.png")

if __name__ == "__main__":
    main()
