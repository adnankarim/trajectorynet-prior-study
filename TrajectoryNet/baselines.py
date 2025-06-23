# -----------------------------------------------------------------------------
# baselines.py      (drop-in replacement)
# -----------------------------------------------------------------------------
import numpy as np
import scipy.spatial
import argparse, textwrap
from TrajectoryNet.dataset import SCData
from TrajectoryNet.optimal_transport.emd import earth_mover_distance
from TrajectoryNet.dataset import interpolate_with_ot


def _sample(arr, n, rng):
    """random sample WITH replacement – returns None if array empty"""
    if len(arr) == 0:
        return None
    idx = rng.choice(len(arr), n, replace=True)
    return arr[idx]


def compute_baselines(ds, leaveout_tp, n_samples=5000, seed=42):
    """
    Returns a *flat* dict:
        'EMD_prev' → float | None
        'EMD_next' → ...
        'MSE_prev' → ...
        ...
    Missing neighbours are reported as None.
    """
    rng    = np.random.RandomState(seed)
    X      = ds.get_data()
    labels = ds.get_times()

    X_hold = _sample(X[labels ==  leaveout_tp     ], n_samples, rng)
    X_prev = _sample(X[labels == (leaveout_tp-1) ], n_samples, rng)
    X_next = _sample(X[labels == (leaveout_tp+1) ], n_samples, rng)
    X_rand = _sample(X[ rng.choice(len(X), n_samples) ], n_samples, rng)

    # OT (McCann) interpolant between prev and next ---------------------------
    X_ot = None
    if X_prev is not None and X_next is not None:
        import ot
        G  = ot.emd2(
            np.ones(len(X_prev))/len(X_prev),
            np.ones(len(X_next))/len(X_next),
            scipy.spatial.distance.cdist(X_prev, X_next)
        )
        X_ot = interpolate_with_ot(X_prev, X_next, G, 0.5, n_samples)

    # helpers -----------------------------------------------------------------
    def emd(a,b): return None if a is None or b is None else earth_mover_distance(a,b)
    def mse(a,b): return None if a is None or b is None else np.mean((a-b)**2)

    out = {
        'EMD_prev': emd(X_prev, X_hold),
        'EMD_next': emd(X_next, X_hold),
        'EMD_rand': emd(X_rand, X_hold),
        'EMD_OT'  : emd(X_ot,   X_hold),
        'MSE_prev': mse(X_prev, X_hold),
        'MSE_next': mse(X_next, X_hold),
        'MSE_rand': mse(X_rand, X_hold),
        'MSE_OT'  : mse(X_ot,   X_hold)
    }
    return out


# -----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent("""\
                Compute simple baselines (prev/next/rand/OT) for a synthetic
                dataset and print a small table. Missing neighbour snapshots
                are reported as '–'.
            """))
    p.add_argument('--dataset', default='TREE')
    p.add_argument('--leaveout', type=int, default=1)
    p.add_argument('--samples',  type=int, default=5000)
    args = p.parse_args()

    ds = SCData.factory(args.dataset, argparse.Namespace(embedding_name='pca',
                                                         max_dim=10,
                                                         whiten=False))

    res = compute_baselines(ds, args.leaveout, args.samples)

    # ------------------------------------------------------------------ print
    print(f"\nBaselines for {args.dataset}, leave-out tp={args.leaveout}\n")
    head = ('', 'prev', 'next', 'rand', 'OT')
    rows = [
        ('EMD',
         res['EMD_prev'], res['EMD_next'], res['EMD_rand'], res['EMD_OT']),
        ('MSE',
         res['MSE_prev'], res['MSE_next'], res['MSE_rand'], res['MSE_OT'])
    ]

    col_w = 10
    print(''.join(f"{h:>{col_w}}" for h in head))
    for name, *vals in rows:
        line = f"{name:>{col_w}}"
        for v in vals:
            line += f"{'–' if v is None else f'{v:.4f}':>{col_w}}"
        print(line)
    print()


if __name__ == '__main__':
    main()
