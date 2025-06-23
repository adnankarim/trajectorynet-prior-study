# TrajectoryNet/dataset.py  (excerpt)
# ------------------------------------------------------------
#  Extended PriorMixin with Gaussian, GMM, Empirical and NSF priors
#  – now with *best-checkpoint* tracking for the NSF prior
# ------------------------------------------------------------

from pathlib import Path
import os, math, joblib, torch, numpy as np
from sklearn.mixture import GaussianMixture

# Optional – only needed for the NSF prior
try:
    import nflows
    from nflows.distributions import StandardNormal
    from nflows.flows import Flow
    from nflows.transforms import (
        CompositeTransform,
        ReversePermutation,
        ActNorm,
    )
    from nflows.transforms.autoregressive import (
        MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    )
except ImportError:
    nflows = None  # we’ll raise if the NSF prior is requested

timepoint=0
class PriorMixin:
    """Mixin that plugs a base‐density sampler / logp into SCData."""

    # ------------------------------------------------------------------
    # 1. Standard isotropic Gaussian 𝓝(0,I)
    # ------------------------------------------------------------------
    def _set_gaussian_prior(self):
        logZ = -0.5 * math.log(2 * math.pi)

        def _logp(z):  # (N,D) → (N,1)
            return torch.sum(logZ - 0.5 * z.pow(2), dim=1, keepdim=True)

        def _sample(n, d, device=None):
            return torch.randn(n, d, device=device or "cpu")

        self.base_density = lambda: _logp
        self.base_sample   = lambda: _sample
    
    def _set_gmmytorch_prior(self, args):
        """
        Build p_u(z1) = (1/K) * Σ_k 𝓝(z1 | μ_k, I),
        where μ_k is the empirical mean of mode / class k.
        """
        device = torch.device( (
        "cuda" if torch.cuda.is_available() else "cpu"))

    
        cache_path = Path(
            getattr(args, "gmm_path", "priors/{dataset}_emp_gmm.pt")
        ).with_suffix(".pt").resolve()
        cache_path = Path(str(cache_path).format(dataset=args.dataset))

        # ------------------------------------------------------------------
        # Step 0:  grab raw data and mode labels
        # ------------------------------------------------------------------
        x0     = torch.tensor(self.get_data(), dtype=torch.float32, device=device)
        labels = torch.tensor(self.labels,     dtype=torch.long,   device=device)
        K, D   = int(labels.max().item() + 1), x0.shape[1]

        # ------------------------------------------------------------------
        # Step 1: load (or compute & save) the mixture parameters
        # ------------------------------------------------------------------
        if cache_path.is_file():
            ckpt = torch.load(cache_path)
            means = ckpt["means"].to(device)
            print("means loaded")
        else:
            means = torch.zeros(K, D)
            for k in range(K):
                means[k] = x0[labels == k].mean(0)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"means": means}, cache_path)
            means=means.to(device)
            print("means saved")
        # ------------------------------------------------------------------
        # Step 2: build a MixtureSameFamily  (uniform weights, I-covariance)
        # ------------------------------------------------------------------
        cat  = torch.distributions.Categorical(
            logits=torch.zeros(K, device=means.device)
        )
        comp = torch.distributions.Independent(
            torch.distributions.Normal(means, torch.ones_like(means)), 1
        )
        gmm = torch.distributions.MixtureSameFamily(cat, comp)

        # ------------------------------------------------------------------
        # Step 3: expose sampler + log-prob in the PriorMixin interface
        # ------------------------------------------------------------------
        def _logp(z):                       # (N,D) → (N,1)
            return gmm.log_prob(z).unsqueeze(1)

        def _sample(n, _d_unused, device=None):
            print("_set_gmmytorch_prior called")
            return gmm.sample((n,)).to(device or "cpu")

        self.base_density = lambda: _logp
        self.base_sample  = lambda: _sample

        # ------------------------------------------------------------------
        # 3. Empirical prior – bootstrap directly from the earliest snapshot
        # ------------------------------------------------------------------
        def _set_empirical_prior(self, _args):
            t0 = sorted(self.get_unique_times())[0]
            buf = torch.tensor(self.get_data()[self.get_times() == t0], dtype=torch.float32)

            def _sample(n, _d, device=None):
                idx = torch.randint(0, buf.shape[0], (n,), device=device or "cpu")
                return buf[idx].to(device or "cpu")

            def _logp(z):
                return torch.zeros(z.shape[0], 1, device=z.device)  # uniform

            self.base_sample  = lambda: _sample
            self.base_density = lambda: _logp

    # ------------------------------------------------------------------
    # 2. Gaussian Mixture Model fitted on the first time-point (t=0)
    # ------------------------------------------------------------------
    def _set_gmm_prior(self, args):
        path = Path(args.gmm_path.format(dataset=args.dataset))
        if path.is_file():
            gmm = joblib.load(path)
        else:
            t0 = self.get_data()[self.get_times() == timepoint]
            gmm = GaussianMixture(
                n_components=14,
                covariance_type="full",
                random_state=42,
            ).fit(t0)
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(gmm, path)

        def _logp(z):
            lp = gmm.score_samples(z.detach().cpu().numpy())
            return torch.from_numpy(lp).to(z).unsqueeze(1)

        def _sample(n, d_unused, device=None):
            x, _ = gmm.sample(n)
            return torch.from_numpy(x.astype(np.float32)).to(device or "cpu")

        self.base_density = lambda: _logp
        self.base_sample   = lambda: _sample

    # ------------------------------------------------------------------
    # 3. Empirical prior – bootstrap directly from the earliest snapshot
    # ------------------------------------------------------------------
    def _set_empirical_prior(self, _args):
        t0 = sorted(self.get_unique_times())[timepoint]
        buf = torch.tensor(self.get_data()[self.get_times() == t0], dtype=torch.float32)

        def _sample(n, _d, device=None):
            idx = torch.randint(0, buf.shape[0], (n,), device=device or "cpu")
            return buf[idx].to(device or "cpu")

        def _logp(z):
            return torch.zeros(z.shape[0], 1, device=z.device)  # uniform

        self.base_sample  = lambda: _sample
        self.base_density = lambda: _logp

       # ------------------------------------------------------------------
    # 4. Neural Spline Flow (NSF) prior – expressive, invertible
    #    with best‐checkpoint tracking (state_dict only)
    # ------------------------------------------------------------------
    def _set_nsf_prior(self, args):
        if nflows is None:
            raise ImportError(
                "NSF prior requested but the `nflows` package is not installed.\n"
                "Install it via `pip install nflows` or choose another prior."
            )

        nsf_path = Path(getattr(
            args, "nsf_path", f"priors/{args.dataset}_nsf.pt"
        ).format(dataset=args.dataset))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = self.get_shape()[0]

        # -- compute whitening stats on raw t=0 once --
        raw_t0 = self.get_data()[self.get_times() == timepoint]
        t0 = torch.tensor(raw_t0, dtype=torch.float32, device=device)
        mean0 = t0.mean(0, keepdim=True)
        std0  = t0.std(0, keepdim=True)

        def _build_flow():
            n_layers = getattr(args, "nsf_layers", 8)
            hidden   = getattr(args, "nsf_hidden", 128)
            tail_b   = getattr(args, "nsf_bound", 15.0)
            blocks = []
            for _ in range(n_layers):
                blocks.extend([
                    ActNorm(features=d),
                    ReversePermutation(features=d),
                    MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                        features=d,
                        hidden_features=hidden,
                        num_bins=8,
                        tails="linear",
                        tail_bound=tail_b,
                    ),
                ])
            return Flow(CompositeTransform(blocks), StandardNormal([d])).to(device)

        # 1) load existing checkpoint if present
        if nsf_path.is_file():
            flow = _build_flow()
            sd = torch.load(nsf_path, map_location=device)
            flow.load_state_dict(sd)
        else:
            # 2) train from scratch, track best‐loss
            flow   = _build_flow()
            white  = (t0 - mean0) / std0
            optim  = torch.optim.Adam(flow.parameters(), lr=1e-3)
            n_ep   = getattr(args, "nsf_epochs", 1000)
            bs     = min(1024, len(white))
            best_l = float("inf")
            best_sd = None

            for ep in range(n_ep):
                idx  = torch.randint(0, len(white), (bs,), device=device)
                x    = white[idx]
                loss = -flow.log_prob(x).mean()
                if torch.isnan(loss):
                    raise RuntimeError("NSF diverged (NaN). Lower lr/epochs.")
                if loss.item() < best_l:
                    best_l  = loss.item()
                    best_sd = {k: v.cpu() for k, v in flow.state_dict().items()}
                optim.zero_grad(); loss.backward(); optim.step()
                if ep % 100 == 0:
                    print(f"[NSF] epoch {ep}/{n_ep}  loss={loss:.3g}")

            # save & restore best
            nsf_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_sd, nsf_path)
            flow.load_state_dict(best_sd)

        flow.eval()

        # expose sampler & log‐prob in original space
        def _logp(z):
            with torch.no_grad():
                # z: raw → whiten → log_prob
                w = (z.to(device) - mean0) / std0
                return flow.log_prob(w).unsqueeze(1)

        def _sample(n, _d_unused, device=None):
            with torch.no_grad():
                w = flow.sample(n).to(device or "cpu")
                z = w * std0.cpu() + mean0.cpu()  # un-whitened sample

                # Compute z-score
                zscore = torch.abs((z - mean0.cpu()) / (std0.cpu() + 1e-6))

                # Find outliers: any dimension > 4 stds
                outlier_mask = (zscore > 4).any(dim=1)

                # Replace outliers with mean vector
                z[outlier_mask] = mean0.cpu().repeat(outlier_mask.sum(), 1)

                return z

        self.base_density = lambda: _logp
        self.base_sample  = lambda: _sample


    # ------------------------------------------------------------------
    # 5. Front-end selector  ------------------------------------------------
    # ------------------------------------------------------------------
    def configure_prior(self, args):
        if   args.prior == "gaussian":
            self._set_gaussian_prior()
        elif args.prior == "gmm":
            self._set_gmm_prior(args)
        elif args.prior == "gmmt":
            self._set_gmmytorch_prior(args)
        elif args.prior == "empirical":
            self._set_empirical_prior(args)
        elif args.prior == "nsf":
            self._set_nsf_prior(args)
        else:
            raise NotImplementedError(f"Unknown prior type: {args.prior}")
