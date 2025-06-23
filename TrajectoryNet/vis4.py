import matplotlib.pyplot as plt
# ----------------------------------------------------------------------
#  Paper-faithful Arch dataset
#  • 3 time-points:   0  →  ½  →  1
#  • 5000 samples at each time-point
# ----------------------------------------------------------------------

import numpy as np, torch, math
from TrajectoryNet.dataset import SCData, TreeTestDataPaper
import ot  # POT: "pip install pot" if missing
class ArchDatasetPaper(SCData):
    """
    Synthetic “arch” data set exactly as described in the TrajectoryNet paper
    (Fig. 4).  Three snapshots on a half-circle of radius ≈1.

        tp=0 : left foot   (θ≈π,  x≈-1, y≈0)
        tp=1 : top centre (θ≈π/2, x≈ 0, y≈1)
        tp=2 : right foot  (θ≈0,  x≈+1, y≈0)
    """

    def __init__(self, n_per_tp: int = 5000, seed: int = 42):
        super().__init__()
        rng = np.random.default_rng(seed)

        # 1) 1-D half-Gaussians, σ = 1/(2π)
        sigma = 1.0 / (2 * math.pi)
        t0 = np.abs(rng.normal(0.0, sigma, n_per_tp))   # tp 0
        t2 = np.abs(rng.normal(1.0, sigma, n_per_tp))   # tp 2

        # 2) McCann interpolant for tp 1
        gamma = ot.emd_1d(t0, t2)
        t1 = self._interpolate_ot(t0, t2, gamma, 0.5, n_per_tp)

        t_all = np.concatenate([t0, t1, t2])
        self.labels = np.repeat(np.arange(3), n_per_tp)          # 0,1,2

        # 3) Lift onto half-circle
        theta = t_all * math.pi                                  # ∈[0,π]
        #   add radius noise: 0.10 for tp 0-1,   0.05 for tp 2
        r_noise = np.full_like(theta, 0.10)
        r_noise[self.labels == 2] = 0.1
        r = 1.0 + rng.normal(0, r_noise)

        x = np.cos(theta) * r
        y = np.sin(theta) * r
        y[self.labels == 2] = np.maximum(y[self.labels == 2], 0.0)  # clamp

        self.data   = np.stack([x, y], axis=1).astype(np.float32)
        self.ncells = self.data.shape[0]

        # 4) analytic, unit-norm velocity (tangent to the circle)
        v = np.stack([-np.sin(theta), np.cos(theta)], axis=1)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        self.velocity = v.astype(np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def _interpolate_ot(p0, p1, tmap, frac, n_out):
        """McCann interpolant (1-D OT map) at fraction *frac*."""
        p = tmap / np.power(tmap.sum(axis=0), 1.0 - frac)
        p = p.ravel() / p.sum()
        I, J = len(p0), len(p1)
        idx  = np.random.choice(I * J, size=n_out, p=p)
        return p0[idx // J] * (1 - frac) + p1[idx % J] * frac

    # ------------  TrajectoryNet API ---------------------------------
    def known_base_density(self):  return True
    def base_density(self):
        logZ = -0.5 * math.log(2 * math.pi)
        return lambda z: torch.sum(logZ - 0.5 * z.pow(2), 1, keepdim=True)
    def base_sample(self):         return torch.randn

    def get_data(self):            return self.data
    def get_times(self):           return self.labels
    def get_unique_times(self):    return np.unique(self.labels)
    def get_velocity(self):        return self.velocity
    def has_velocity(self):        return True
    def get_ncells(self):          return self.ncells
    def get_shape(self):           return [self.data.shape[1]]
    def sample_index(self, n, tp):
        idx = np.where(self.labels == tp)[0]
        return np.random.choice(idx, size=n)

def main():
    ds = TreeTestDataPaper()
    X = ds.get_data()
    y = ds.get_times()

    print("data shape        :", X.shape)
    print("labels unique     :", np.unique(y, return_counts=True))
    print("min/max x,y       :", X.min(axis=0), X.max(axis=0))
    print("velocity present? :", ds.has_velocity())

    # Fix colormap issue
    unique_labels = [int(i) for i in np.unique(y)]
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    cmap = {int(tp): color for tp, color in zip(unique_labels, colors)}

    plt.figure(figsize=(6, 6))
    for tp in unique_labels:
        m = (y == tp)
        plt.scatter(X[m, 0], X[m, 1], s=4, alpha=0.6, c=cmap[tp], label=f"tp={tp}")

    plt.legend()
    plt.axis('equal')
    plt.title("TREEDatasetPaper")
    plt.savefig("tree_dataset.png", dpi=300)  # Save image
    print("Saved plot to tree_dataset.png")

if __name__ == "__main__":
    main()
