#!/usr/bin/env python
# ------------------------------------------------------------
#  prior_sanity.py
#
#  • Loads one of TrajectoryNet’s priors  (gaussian | gmm | empirical | nsf)
#  • Draws quick overlays against the earliest snapshot (t = 0)
#  • Caches the fitted prior (GMM / NSF) so you can reuse it in training
# ------------------------------------------------------------
import argparse, math, pathlib, sys
import torch, matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# TrajectoryNet imports -------------------------------------------------------
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from TrajectoryNet import dataset                           # noqa: 402
from TrajectoryNet.mixins import PriorMixin                 # noqa: 402

# -----------------------------------------------------------------------------


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visual check of a chosen base prior in 2-D datasets."
    )
    p.add_argument("--dataset", default="CYCLE", help="any of the SCData demos")
    p.add_argument(
        "--prior",
        default="gaussian",
        choices=["gaussian", "gmm", "empirical", "nsf"],
        help="which base density to test",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="number of samples to draw from the prior for the overlay",
    )
    p.add_argument(
        "--savefig",
        default="png",
        help="optional path to save the figure (png/pdf). If omitted, just shows.",
    )
    # hyper-params that only affect GMM / NSF; ignored otherwise
    p.add_argument("--gmm_path", default="priors/{dataset}_t0_gmm.pkl")
    p.add_argument("--nsf_path", default="priors/{dataset}_nsf.pkl")
    p.add_argument("--nsf_epochs", type=int, default=1000)
    p.add_argument("--nsf_layers", type=int, default=8)
    p.add_argument("--nsf_hidden", type=int, default=128)
    p.add_argument("--nsf_bound", type=float, default=15.0)
    return p


@torch.no_grad()
def main(opts: argparse.Namespace):

    # ---------- 1  load raw data ----------------
    scdata = dataset.SCData.factory(opts.dataset, opts)
   # SCData already has configure_prior(), only patch if it doesn’t
    if not hasattr(scdata, "configure_prior"):
        scdata.__class__ = type("SCDataPrior", (PriorMixin, scdata.__class__), {})


    # ---------- 2  configure / (fit) the chosen prior ---------------
    scdata.configure_prior(opts)

    # ---------- 3  draw samples & ground-truth points ---------------
    d = scdata.get_shape()[-1]
    if d != 2:
        print(
            f"[WARN] visual check designed for 2-D embeddings, "
            f"but your data has dim={d}.  "
            "Nothing plotted."
        )
        sys.exit(0)

    # earliest snapshot (usually t = 0)
    t0 = sorted(scdata.get_unique_times())[-1]
    data_t0 = scdata.get_data()[scdata.get_times() == t0]

    z = scdata.base_sample()(opts.samples, d).cpu().numpy()

    # ------------- 4  pretty scatter plot ---------------------------
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    ax.scatter(
        data_t0[:, 0],
        data_t0[:, 1],
        s=6,
        alpha=0.35,
        label=f"t={t0} data ({len(data_t0)})",
        color="#1f77b4",
    )
    ax.scatter(
        z[:, 0],
        z[:, 1],
        s=6,
        alpha=0.35,
        label=f"{opts.prior} prior ({opts.samples})",
        color="#ff7f0e",
    )
    ax.set_aspect("equal")
    ax.set_title(f"{opts.dataset} – sanity check of '{opts.prior}' base density")
    ax.legend(frameon=False)
    sns.despine()

    if opts.savefig:
        pathlib.Path(opts.savefig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(opts.savefig, bbox_inches="tight")
        print(f"✅  saved figure → {opts.savefig}")
    else:
        plt.show()


if __name__ == "__main__":
    main(make_parser().parse_args())
