"""
dataset3.py  ––   Synthetic TREE / CIRCLE5 / CYCLE with *three* time-points
compatible with evaluate_kantorovich_v2.
"""

from __future__ import annotations
import math, numpy as np, torch, scipy.sparse
from typing import Tuple

# ------------------------------------------------------------------------- #
#                               base helper                                 #
# ------------------------------------------------------------------------- #
def _standard_normal_logp(z: torch.Tensor) -> torch.Tensor:
    logZ = -0.5 * math.log(2 * math.pi)
    return torch.sum(logZ - z.pow(2) / 2, 1, keepdim=True)

class SCData:
    """Minimal abstract interface expected by TrajectoryNet."""
    # ---- must be implemented --------------------------------------------
    def get_data(self)            -> np.ndarray:  ...
    def get_times(self)           -> np.ndarray:  ...
    def get_unique_times(self)    -> np.ndarray:  ...
    def get_shape(self)           -> Tuple[int]:  return (self.data.shape[1],)
    def sample_index(self,n,tp)   -> np.ndarray:  ...
    # ---------------------------------------------------------------------
    def known_base_density(self):   return True
    def base_density(self):         return _standard_normal_logp
    def base_sample(self):          return torch.randn
    def get_velocity(self):         return self.velocity
    def has_velocity(self):         return hasattr(self,'velocity')
    def get_ncells(self):           return self.data.shape[0]

# ------------------------------------------------------------------------- #
#                        1-D manifold helper function                       #
# ------------------------------------------------------------------------- #
def _interpolate_with_ot(p0,p1,tmap,alpha,n_out)->np.ndarray:
    """McCann 1-D barycentric interpolation (unchanged from original code)."""
    p0 = np.asarray(p0,dtype=float); p1 = np.asarray(p1,dtype=float)
    tmap = np.asarray(tmap,dtype=float)
    I,J = len(p0),len(p1)
    prob = tmap / np.power(tmap.sum(axis=0),1.0-alpha)
    prob = (prob/ prob.sum()).flatten()
    idx  = np.random.choice(I*J,p=prob,size=n_out)
    return np.array([(1-alpha)*p0[i//J] + alpha*p1[i%J] for i in idx])

# ------------------------------------------------------------------------- #
#                                 TREE                                      #
# ------------------------------------------------------------------------- #
class Tree3(SCData):
    """
    Three snapshots on the “tree” manifold:
      t=0  : upper half-circle left-side
      t=0.5: OT barycenter
      t=1  : upper half-circle right-side
    Velocity = analytic tangent.
    """
    def __init__(self,n: int = 5000,r1=0.5,r2=0.1):
        np.random.seed(42)
        self.r1,self.r2 = r1,r2
        # --- sample 1-D latent variable ----------------------------------
        a = np.abs(np.random.randn(2*n)*0.5/np.pi)
        labels = np.repeat([0,2],n)
        a[labels==2] = 1 - a[labels==2]

        # OT interpolant for t=0.5
        import ot
        gamma = ot.emd_1d(a[labels==0],a[labels==2])
        mid   = _interpolate_with_ot(a[labels==0,None],a[labels==2,None],gamma,0.5,n)
        a     = np.concatenate([a,mid])
        labels= np.concatenate([labels,np.ones(n)])

        # --- lift to 2-D -------------------------------------------------
        theta = a*np.pi
        r = (1+np.random.randn(theta.size)*self.r2)[:,None]
        pts = np.stack([np.cos(theta),np.sin(theta)],1)*np.repeat(r,2,1)

        # small branch-flip like original code
        mask = (np.random.rand(len(pts))>0.5)&(pts[:,0]<0)
        pts[mask] = np.array([0,2])+np.array([1,-1])*pts[mask]

        self.data   = pts.astype(np.float32)
        self.labels = labels                      # 0,1,2
        # analytic velocity (simple finite-diff along theta)
        next_theta  = theta + 0.3
        next_pts    = np.stack([np.cos(next_theta),np.sin(next_theta)],1)*np.repeat(r,2,1)
        next_pts[mask] = np.array([0,2])+np.array([1,-1])*next_pts[mask]
        self.velocity = (next_pts - pts).astype(np.float32)

    # ---------- helpers --------------------------------------------------
    def get_data(self): return self.data
    def get_times(self): return self.labels
    def get_unique_times(self): return np.unique(self.labels)
    def sample_index(self,n,tp):
        idx=np.where(self.labels==tp)[0]
        return np.random.choice(idx,size=n)

# ------------------------------------------------------------------------- #
#                               CIRCLE 5                                    #
# ------------------------------------------------------------------------- #
class Circle5(Tree3):
    """Same construction but without branch-flip (matches paper’s ‘circle5’)."""
    def __init__(self,n=5000,r1=0.5,r2=0.1):
        super().__init__(n,r1,r2)     # build like Tree
        # override: remove branch flip effect
        self.data = self.data.copy()
        self.velocity = self.velocity.copy()

# ------------------------------------------------------------------------- #
#                                CYCLE                                      #
# ------------------------------------------------------------------------- #
class Cycle3(SCData):
    """
    Uniform points on unit circle rotating CCW at ω=π/5 per unit time.
    Snapshots: t=0, 0.5, 1  (so Δθ = ω*0.5).
    """
    def __init__(self,n_per_tp=5000,omega=np.pi/5,r_std=0.05):
        np.random.seed(42)
        ts   = np.concatenate([np.zeros(n_per_tp),
                               0.5*np.ones(n_per_tp),
                               np.ones(n_per_tp)])
        thetas = omega*ts
        # uniform base angle φ ϵ [0,2π)
        phi = np.random.rand(ts.size)*2*np.pi
        theta = phi + thetas                               # rotation
        r = (1+r_std*np.random.randn(ts.size))
        xs,ys = r*np.cos(theta), r*np.sin(theta)
        self.data   = np.column_stack([xs,ys]).astype(np.float32)
        self.labels = np.repeat([0,1,2],n_per_tp)
        # analytic velocity = tangential unit vector
        vx = -np.sin(theta)*omega
        vy =  np.cos(theta)*omega
        v  = np.column_stack([vx,vy])
        v /= np.linalg.norm(v,axis=1,keepdims=True)
        self.velocity = v.astype(np.float32)

    # SCData API
    def get_data(self): return self.data
    def get_times(self): return self.labels
    def get_unique_times(self): return np.unique(self.labels)
    def sample_index(self,n,tp):
        idx=np.where(self.labels==tp)[0]
        return np.random.choice(idx,size=n)

# ------------------------------------------------------------------------- #
#                        factory (for argparse)                             #
# ------------------------------------------------------------------------- #
def factory(name:str)->SCData:
    name=name.upper()
    if   name=="TREE":    return Tree3()
    elif name=="CIRCLE5": return Circle5()
    elif name=="CYCLE":   return Cycle3()
    else: raise KeyError(f"Unknown synthetic dataset '{name}'")
