# -------------------------------------------------------------------------
#  viz_baselines.py  –  visualise static baselines (prev / next / rand / OT)
# -------------------------------------------------------------------------
#  Author:  <you>                                       Licence: MIT
# -------------------------------------------------------------------------
"""
Generate the same kind of pictures / movies that `plot_output()` produces
for a trained CNF – but here for the four *static* baselines that are
used in baseline-bio.py (prev / next / rand / OT midpoint).

Run e.g.

    python -m TrajectoryNet.viz_baselines \
           --dataset EB --leaveout_timepoint 2 --save results/bio_vis

All usual TrajectoryNet CLI flags work; only --leaveout_timepoint must be
1, 2 or 3.
"""
import os, warnings, numpy as np, torch, matplotlib.pyplot as plt
import ot                                  # POT
from TrajectoryNet import dataset
from TrajectoryNet.parse import parser
from TrajectoryNet.lib.utils import makedirs
# ---------------------------------------------------------------------
#  save_vectors – now works with model *or* plain OT
# ---------------------------------------------------------------------
import os, numpy as np, torch, matplotlib.pyplot as plt, matplotlib
import ot                                            # POT
from TrajectoryNet.lib.utils import makedirs         # already in the repo


def _ot_barycentric(source, target, reg=5e-2):
    """
    Return barycentric OT map  T:X→Y  for two point clouds
    using entropic Sinkhorn.  Both clouds must be (n,d) / (m,d) numpy.
    """
    n, m = len(source), len(target)
    a = np.full(n, 1 / n)
    b = np.full(m, 1 / m)
    C = ot.dist(source, target)          # (n,m) cost matrix
    G = ot.sinkhorn(a, b, C, reg=reg)    # transport plan
    # barycentric projection:  T(x_i) = Σ_j γ_ij · y_j  /  Σ_j γ_ij
    weights = G / G.sum(axis=1, keepdims=True)
    return weights @ target              # (n,d) mapped positions


def save_vectors(prior_logdensity,
                 model,                     # may be None !
                 data_samples,              # torch (n,d)
                 full_data,                 # np (N,d) – “future”
                 labels,                    # np (N,)
                 savedir,
                 skip_first=False,
                 ntimes=101,
                 end_times=None,
                 memory=0.01,
                 device='cpu',
                 lim=4,
                 sinkhorn_eps=5e-2):
    """
    Draw little trajectories (“vectors”) starting from `data_samples`.
    When `model` is None we fall back to a *static* OT barycentric map.
    """
    os.makedirs(savedir, exist_ok=True)
    z0 = data_samples.detach().cpu().numpy()        # (n,d)

    # ------------------------------------------------------------------
    # 1. obtain the *end* positions z1
    # ------------------------------------------------------------------
    if model is None:                               # ---- plain OT ----
        z1 = _ot_barycentric(z0, full_data, reg=sinkhorn_eps)

    else:                                           # ---- CNF model ----
        model.eval()
        with torch.no_grad():
            z = data_samples.to(device)
            logp = prior_logdensity(z)
            cnf  = model.chain[0]                   # single CNF block

            if end_times is None:
                end_times = [cnf.sqrt_end_time ** 2]

            int_list = [torch.linspace(0, end_times[0], ntimes,
                                        device=device)]
            for i, et in enumerate(end_times[1:]):
                int_list.append(torch.linspace(end_times[i], et, ntimes,
                                                device=device))

            # integrate once – we only need the *first* step for arrows
            z_traj, _ = cnf(z, logp, integration_times=int_list[0],
                            reverse=True)
            z1 = z_traj[:, -1, :].cpu().numpy()     # (n,d)

    # ------------------------------------------------------------------
    # 2. plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    # 2-a  full background cloud
    pos_mask = full_data[:, 1] >= 0           # keep only y≥0 (paper habit)
    ax.scatter(full_data[pos_mask, 0],
               full_data[pos_mask, 1],
               c=labels[pos_mask].astype(int),
               cmap='tab10', s=0.5, alpha=1.0)

    # 2-b  starting points
    ax.scatter(z0[:, 0], z0[:, 1], c='k', s=10, zorder=5)

    # 2-c  arrows
    for p0, p1 in zip(z0, z1):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                linewidth=0.8, color='crimson', alpha=0.7)

    ax.set_xlim(-lim, lim);  ax.set_ylim(-lim, lim)
    ax.set_xticks([]);       ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "vectors.jpg"), dpi=300)
    plt.close(fig)

# ------------------- parameters identical to baseline-bio.py ------------------
SINKHORN_EPS = 1e-1
MAX_POINTS   = 5000
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------#
# helper: barycentric midpoint via Sinkhorn                                    #
# -----------------------------------------------------------------------------#
def midpoint_ot(prev, nxt, k=MAX_POINTS, rng=np.random):
    """McCann midpoint with importance subsampling."""
    a = np.full(len(prev), 1 / len(prev))
    b = np.full(len(nxt),  1 / len(nxt))
    G = ot.sinkhorn(a, b, ot.dist(prev, nxt), reg=SINKHORN_EPS)
    rows, cols = np.nonzero(G)
    mass  = G[rows, cols]
    probs = mass / mass.sum()
    take  = min(k, len(probs))
    sel   = rng.choice(len(probs), take, replace=False, p=probs)
    return 0.5 * (prev[rows[sel]] + nxt[cols[sel]])


# -----------------------------------------------------------------------------#
# cheap stand-ins for save_trajectory* when model=None                         #
# -----------------------------------------------------------------------------#
def _quick_trajectory(cloud, out_dir, lim=4):
    """Draw little ‘pearls on a string’ between start-points and cloud."""
    makedirs(out_dir)
    # sample random subset for clarity
    if len(cloud) > 2000:
        cloud = cloud[np.random.choice(len(cloud), 2000, replace=False)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.scatter(cloud[:, 0], cloud[:, 1],
               s=4, c=np.linspace(0, 1, len(cloud)), cmap="Spectral")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "traj_00.jpg"), dpi=300)
    plt.close(fig)


def _quick_density(cloud, out_dir, lim=4):
    """Single PNG with 2-D KDE; enough for movie placeholder."""
    from scipy.stats import gaussian_kde
    makedirs(out_dir)
    if len(cloud) > 5000:
        cloud = cloud[np.random.choice(len(cloud), 5000, replace=False)]
    kde = gaussian_kde(cloud.T)

    side = np.linspace(-lim, lim, 200)
    xx, yy = np.meshgrid(side, side)
    z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 6))
    pcm = ax.pcolormesh(xx, yy, z, cmap="magma")
    fig.colorbar(pcm, ax=ax, shrink=0.8)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "density_00.jpg"), dpi=300)
    plt.close(fig)



# -----------------------------------------------------------------------------#
# main helpers                                                                 #
# -----------------------------------------------------------------------------#
def make_prediction_clouds(sc_data, leave_out, rng):
    """Return dict baseline → numpy cloud that ‘predicts’ the left-out slice."""
    times   = sc_data.get_unique_times()
    X_prev  = sc_data.get_data()[sc_data.get_times() == times[leave_out - 1]]
    X_mid   = sc_data.get_data()[sc_data.get_times() == times[leave_out]]
    X_next  = sc_data.get_data()[sc_data.get_times() == times[leave_out + 1]]

    return {
        "prev": X_prev,
        "next": X_next,
        "rand": X_prev if rng.rand() < 0.5 else X_next,
        "OT"  : midpoint_ot(X_prev, X_next, rng=rng),
    }


def visualise_baseline(bname, cloud, args, save_root):
    """Replicates plot_output() but for a static cloud (no model)."""
    out_dir = os.path.join(save_root, bname)
    os.makedirs(out_dir, exist_ok=True)

    # ---------- little arrows (save_vectors with model=None) -----------------
    np.random.seed(42)
    start_pts = args.data.base_sample()(1000, cloud.shape[1])
    save_vectors(
        args.data.base_density(),
        None,                           # ← no model
        torch.tensor(start_pts, dtype=torch.float32),
        cloud,
        np.full(len(cloud), args.leaveout_timepoint),
        out_dir,
        skip_first=(not args.data.known_base_density()),
        device="cpu",
        end_times=np.array([0, 1]),
        ntimes=2,
    )

    # ---------- quick trajectory + density PNGs -----------------------------
    _quick_trajectory(cloud, os.path.join(out_dir, "trajectory"))
    _quick_density(cloud,    os.path.join(out_dir, "density"))
    # (optional) you could call trajectory_to_video here if you really need MP4


# -----------------------------------------------------------------------------#
def main():
    args = parser.parse_args()

    if args.leaveout_timepoint not in (1, 2, 3):
        raise ValueError("--leaveout_timepoint must be 1, 2 or 3")

    args.data = dataset.SCData.factory(args.dataset, args)

    rng    = np.random.RandomState(0)
    clouds = make_prediction_clouds(args.data, args.leaveout_timepoint, rng)

    save_root = os.path.join(args.save, "baselines_vis")
    os.makedirs(save_root, exist_ok=True)

    print(f"\nWriting baseline visualisations to  {save_root}\n")
    for name, cloud in clouds.items():
        print(f"  • {name}")
        visualise_baseline(name, cloud, args, save_root)


# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        main()
