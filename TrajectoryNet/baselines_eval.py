#!/usr/bin/env python
# -------------------------------------------------------------
#  baselines_eval.py
#  ------------------------------------------------------------
#  Usage examples:
#     python -m  TrajectoryNet.baselines_eval --dataset TREE     --output results/tree_baselines.csv
#     python -m  TrajectoryNet.baselines_eval --dataset CIRCLE5  --output results/circle5_baselines.csv
#     python -m  TrajectoryNet.baselines_eval --dataset CYCLE    --output results/cycle_baselines.csv
#
#  By default it evaluates the "middle" slice (leave-one-out
#  time-point 1).  Change --leaveout if you need another slice.
# -------------------------------------------------------------

import os, argparse, json, random
import numpy as np
import torch
import warnings

# --- TrajectoryNet helpers ----------------------------------------------------
from TrajectoryNet import dataset                # data loader
from TrajectoryNet.eval_utils import earth_mover_distance

# optional – only needed for the OT baseline
try:
    import ot                                    # POT – Python Optimal Transport
except ImportError:
    raise RuntimeError("Please  pip install pot  for the OT baseline")
def normal_mse(x, y):
    """
    Normal MSE between corresponding points of two clouds x and y.
    Assumes x and y are of equal length and aligned.
    """
    x_t = torch.tensor(x)
    y_t = torch.tensor(y)
    min_len = min(len(x_t), len(y_t))
    x_t = x_t[:min_len]
    y_t = y_t[:min_len]
    return torch.mean((x_t - y_t) ** 2).item()

# ------------------------------------------------------------------------------
def chamfer_mse(x, y):
    """
    Symmetric Chamfer distance (mean-squared error) between two
    point clouds  x (N,d)  and  y (M,d)  – a cheap OT-free proxy.
    """
    D = torch.cdist(torch.tensor(x), torch.tensor(y)) ** 2
    mse_xy = D.min(dim=1).values.mean()      # each x to its nearest y
    mse_yx = D.min(dim=0).values.mean()      # each y to its nearest x
    return 0.5 * (mse_xy + mse_yx)

# ------------------------------------------------------------------------------
def midpoint_OT(prev, nxt, numItermax=200000, sinkhorn=False, reg=1e-1):
    """
    Static OT midpoint between two clouds.
      - numItermax: max iterations for the LP solver
      - sinkhorn: if True, use entropic OT (faster and no LP warning)
      - reg: regularization strength for Sinkhorn
    """
    n1, n2 = len(prev), len(nxt)
    w1 = np.ones(n1) / n1
    w2 = np.ones(n2) / n2
    C  = ot.dist(prev, nxt)

    if sinkhorn:
        # entropic regularization → much faster, no LP warning
        G = ot.sinkhorn(w1, w2, C, reg)
    else:
        # classical LP with increased iteration cap
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                message="numItermax reached before optimality")
            G = ot.emd(w1, w2, C, numItermax=numItermax)

    rows, cols = np.nonzero(G)
    return 0.5*(prev[rows] + nxt[cols])
# ------------------------------------------------------------------------------
def evaluate_dataset(name, leaveout_tpi=1, seed=None):
    """
    Returns a dict  {row_name: (emd, mse)}
    for one synthetic data set.
    """
    if seed is not None:
        np.random.seed(seed); random.seed(seed)

    data  = dataset.SCData.factory(name, argparse.Namespace())  # empty args ok
    times = data.get_unique_times()
    t_prev, t_mid, t_next = leaveout_tpi-1, leaveout_tpi, leaveout_tpi+1

    # point clouds
    X_prev = data.get_data()[data.get_times() == times[t_prev]]
    X_mid  = data.get_data()[data.get_times() == times[t_mid]]
    X_next = data.get_data()[data.get_times() == times[t_next]]

    rows = {}

    # ------------ prev ---------------------------------------------------------
    pred = X_prev.copy()
    rows["prev"] = (earth_mover_distance(pred, X_mid),
                    normal_mse(pred, X_mid))

    # ------------ next ---------------------------------------------------------
    pred = X_next.copy()
    rows["next"] = (earth_mover_distance(pred, X_mid),
                    normal_mse(pred, X_mid))

    # ------------ rand ---------------------------------------------------------
    pred = X_prev if random.random() < 0.5 else X_next
    rows["rand"] = (earth_mover_distance(pred, X_mid),
                    normal_mse(pred, X_mid))

    # ------------ OT -----------------------------------------------------------
    pred = midpoint_OT(X_prev, X_next)
    rows["OT"]   = (earth_mover_distance(pred, X_mid),
                    normal_mse(pred, X_mid))

    return rows

# ------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  required=True,
                   choices=["TREE", "CIRCLE5", "CYCLE"],
                   help="Synthetic data set name")
    p.add_argument("--leaveout_timepoint", type=int, default=1,
                   help="Which intermediate slice to hide (default 1)")
    p.add_argument("--seeds", type=int, default=3,
                   help="How many random seeds to average (default 3, "
                        "set to 1 for faster debugging)")
    p.add_argument("--output",  default="runs.csv",
                   help="Path to save a CSV/JSON of the results")
    args = p.parse_args()

    # -------------------------------------------------------- run all seeds ---
    all_rows = {}        # row_name -> list of (emd,mse) tuples
    for s in range(args.seeds):
        res = evaluate_dataset(args.dataset,
                               leaveout_tpi=args.leaveout_timepoint,
                               seed=s)
        for k,v in res.items():
            all_rows.setdefault(k, []).append(v)

    # ---------------------------------------------------- aggregate + print ---
    header = f"**{args.dataset} (leave-out t{args.leaveout_timepoint})**"
    print("\n" + header)
    print("| baseline |  EMD (mean±sd) |  MSE (mean±sd) |")
    print("|----------|---------------:|---------------:|")
    table  = []
    for row in ["OT","prev","next","rand"]:          # fixed order
        vals = np.array(all_rows[row])               # shape (seeds,2)
        emd_mean, mse_mean = vals.mean(0)
        emd_std , mse_std  = vals.std(0)
        print(f"| {row:<4} | {emd_mean:6.4f} ± {emd_std:5.4f}"
              f" | {mse_mean:6.4f} ± {mse_std:5.4f} |")
        table.append([row, emd_mean, emd_std, mse_mean, mse_std])

    # ------------------------------------------------------ optional output ---
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        import csv, json
        if args.output.endswith(".csv"):
            with open(args.output, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["baseline","emd_mean","emd_std",
                            "mse_mean","mse_std"])
                w.writerows(table)
        else:                                       # fall back to JSON
            with open(args.output, "w") as f:
                json.dump({r[0]:r[1:] for r in table}, f, indent=2)
        print(f"\n> saved to {args.output}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
