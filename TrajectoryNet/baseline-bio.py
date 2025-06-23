#!/usr/bin/env python
# ----------------------------------------------------------------------
#  baseline-bio.py   –   memory–safe EB baselines with EMD & (plain) MSE
# ----------------------------------------------------------------------
import os, csv, argparse, warnings
import numpy as np
import torch
import ot                                        # POT – Optimal Transport
from TrajectoryNet import dataset
from TrajectoryNet.parse import parser           # re-use TrajectoryNet’s CLI

# ---------------------- tweak these if you like ----------------------
MAX_POINTS     = 5000       # ≤ 5 000 preserves the paper numbers
SINKHORN_EPS   = 1e-1       # entropic regularisation
SEEDS          = (0, 1, 2)  # three seeds → mean over 3 runs
DATASET_NAME   = "EB"       # change if your alias differs
# ----------------------------------------------------------------------

# ---------- helpers ------------------------------------------------------------
def subsample(X: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    """Return ≤ k rows of X (without replacement)."""
    return X if len(X) <= k else X[rng.choice(len(X), k, replace=False)]

def emd_sinkhorn(X, Y) -> float:
    """Entropic OT (Sinkhorn-2) Wasserstein cost between two clouds."""
    w1 = np.ones(len(X)) / len(X)
    w2 = np.ones(len(Y)) / len(Y)
    C  = ot.dist(X, Y)
    res = ot.sinkhorn2(w1, w2, C, reg=SINKHORN_EPS)
    return res[0] if isinstance(res, (tuple, list, np.ndarray)) else float(res)

def normal_mse(x, y) -> float:
    """Plain MSE between two *aligned* clouds (use equal-length sub-samples)."""
    x_t, y_t = torch.tensor(x), torch.tensor(y)
    n = min(len(x_t), len(y_t))
    return torch.mean((x_t[:n] - y_t[:n]) ** 2).item()

def midpoint_ot(prev, nxt, rng):
    """McCann midpoint via Sinkhorn transport plan – already sub-sampled."""
    n1, n2 = len(prev), len(nxt)
    w1 = np.ones(n1) / n1
    w2 = np.ones(n2) / n2
    G  = ot.sinkhorn(w1, w2, ot.dist(prev, nxt), reg=SINKHORN_EPS)   # (n1,n2)
    rows, cols = np.nonzero(G)
    mass  = G[rows, cols]
    probs = mass / mass.sum()
    take  = min(MAX_POINTS, len(probs))
    sel   = rng.choice(len(probs), take, replace=False, p=probs)
    return 0.5 * (prev[rows[sel]] + nxt[cols[sel]])

# ---------- evaluation of one left-out time slice ------------------------------
def eval_slice(data, leave_out: int, rng):
    """
    leave_out ∈ {1,2,3}.  Returns {'EMD':{..}, 'MSE':{..}} for the four baselines.
    """
    t      = data.get_unique_times()
    X_prev = data.get_data()[data.get_times() == t[leave_out-1]]
    X_mid  = data.get_data()[data.get_times() == t[leave_out]]
    X_next = data.get_data()[data.get_times() == t[leave_out+1]]

    # one fixed sub-sample of the *target* for both metrics
    X_mid_sub = subsample(X_mid, MAX_POINTS, rng)

    # helper closures -----------------------------------------------------------
    def score_emd(pred):
        return emd_sinkhorn(subsample(pred, MAX_POINTS, rng), X_mid_sub)

    def score_mse(pred):
        pred_sub  = subsample(pred, MAX_POINTS, rng)
        n         = min(len(pred_sub), len(X_mid_sub))
        return normal_mse(pred_sub[:n], X_mid_sub[:n])

    # baseline predictions ------------------------------------------------------
    preds = {
         "OT"  : midpoint_ot(X_prev, X_next, rng),
        "prev": X_prev,
        "next": X_next,
        "rand": X_prev if rng.rand() < 0.5 else X_next,
       
    }

    return {"EMD": {k: score_emd(v) for k, v in preds.items()},
            "MSE": {k: score_mse(v) for k, v in preds.items()}}

# ---------- main ---------------------------------------------------------------
def main():
    args  = parser.parse_args()
    data  = dataset.SCData.factory(DATASET_NAME, args)

    baselines = ["prev", "next", "rand", "OT"]
    metrics   = {"EMD": {}, "MSE": {}}
    for m in metrics:
        metrics[m] = {b: {1:[],2:[],3:[]} for b in baselines}

    # loop over seeds and leave-outs -------------------------------------------
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        for lo in (1, 2, 3):
            res = eval_slice(data, lo, rng)
            for m in ("EMD", "MSE"):
                for b in baselines:
                    metrics[m][b][lo].append(res[m][b])

    # pretty print -------------------------------------------------------------
    print(f"\n**EB baselines (Sinkhorn ε={SINKHORN_EPS}, ≤{MAX_POINTS} pts, seeds={SEEDS})**")
    header = ["baseline",
              "EMD_t1","EMD_t2","EMD_t3","EMD_mean",
              "MSE_t1","MSE_t2","MSE_t3","MSE_mean"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---:"]*len(header)) + "|")

    rows_csv = []
    for b in baselines:
        emd = [np.mean(metrics["EMD"][b][t]) for t in (1,2,3)]
        mse = [np.mean(metrics["MSE"][b][t]) for t in (1,2,3)]
        row = [b] + [f"{v:.4f}" for v in emd] + [f"{np.mean(emd):.4f}"] \
                + [f"{v:.4f}" for v in mse] + [f"{np.mean(mse):.4}"]
        print("| " + " | ".join(row) + " |")
        rows_csv.append(row)

    # CSV ----------------------------------------------------------------------
    out_csv = getattr(args, "csv", "bio-eval.csv")
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows([header, *rows_csv])
    print(f"\n> Results written to {out_csv}")

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        main()
