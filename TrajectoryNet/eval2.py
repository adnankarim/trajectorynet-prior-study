"""
eval_v2.py  ––   Evaluate a trained TrajectoryNet on synthetic 3-TP data
computing  (i) Kantorovich v2  and  (ii) MSE against ground-truth paths.
Usage
-----
python eval_v2.py --dataset TREE --ckpt path/to/checkpt.pth --leaveout 1
"""

from __future__ import annotations
import argparse, os, numpy as np, torch, math
from dataset3 import factory as load_dataset           # <-- our new datasets
from TrajectoryNet.train_misc import (build_model_tabular,
                                      set_cnf_options)
from TrajectoryNet.optimal_transport.emd import earth_mover_distance

# --------------------------------------------------------------------- #
def evaluate_kantorovich_v2(model,data,int_tps,leaveout,device):
    """EMD between predicted distribution at left-out tp and truth."""
    assert 0<leaveout<len(int_tps)-1, "leaveout must be middle tp (1)"
    tp_prev, tp_mid, tp_next = leaveout-1, leaveout, leaveout+1

    x_next = torch.from_numpy(data.get_data()[data.get_times()==tp_next]).to(device)
    x_prev = torch.from_numpy(data.get_data()[data.get_times()==tp_prev]).to(device)
    zero   = torch.zeros(len(x_next),1,device=device)

    # integrate back from next to mid
    times = torch.tensor([int_tps[tp_mid],int_tps[tp_next]],dtype=torch.float32,
                         device=device)
    x_pred_b,_ = model.chain[0](x_next,zero,integration_times=times)   # backward

    # integrate fwd from prev to mid
    zero = torch.zeros(len(x_prev),1,device=device)
    times = torch.tensor([int_tps[tp_prev],int_tps[tp_mid]],dtype=torch.float32,
                         device=device)
    x_pred_f,_ = model.chain[0](x_prev,zero,integration_times=times,reverse=True)

    truth = data.get_data()[data.get_times()==tp_mid]
    emd_b = earth_mover_distance(x_pred_b.cpu().numpy(),truth)
    emd_f = earth_mover_distance(x_pred_f.cpu().numpy(),truth)
    return np.array([emd_b,emd_f])

# --------------------------------------------------------------------- #
def evaluate_mse(model,data,int_tps,device,n_paths=5000):
    """Follow n_paths from t0 to t1 via model and compare to ground truth."""
    paths = data.get_paths(n_paths,3)          # shape (N,3,2)
    z0    = torch.tensor(paths[:,0,:],dtype=torch.float32,device=device)
    zero  = torch.zeros(len(z0),1,device=device)
    # integrate to t=0.5 (first interval)
    times = torch.tensor([int_tps[0],int_tps[1]],dtype=torch.float32,device=device)
    z_mid,_ = model.chain[0](z0,zero,integration_times=times,reverse=True)
    mse = np.mean((z_mid.cpu().numpy()-paths[:,1,:])**2,axis=(-1,-2))
    return np.array([mse.mean()])

# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=["TREE","CIRCLE5","CYCLE"],required=True)
    ap.add_argument("--ckpt",required=True,help="path to checkpt.pth")
    ap.add_argument("--leaveout",type=int,default=1,help="time-point to hold-out")
    ap.add_argument("--gpu",type=int,default=-1)
    args = ap.parse_args()

    device = torch.device("cpu" if args.gpu<0 or not torch.cuda.is_available()
                          else f"cuda:{args.gpu}")
    data = load_dataset(args.dataset)
    time_scale = 0.5
    int_tps = (np.arange(3)+1.0)*time_scale          # [0.5,1.0,1.5]

    # ---- load trained model -------------------------------------------
    # model architecture must match training script (dims etc.)
    class DummyArgs:          # minimal stub to reuse builder
        dims="64-64-64"; layer_type="concatsquash"; num_blocks=1
        divergence_fn="brute_force"; residual=False; rademacher=False
        batch_norm=False; bn_lag=0.; atol=1e-5; rtol=1e-5; solver="dopri5"
        time_scale=time_scale; train_T=True; nonlinearity="tanh"
        spectral_norm=False;    # was False in your training cmd
    margs = DummyArgs()
    model = build_model_tabular(margs,2,regularization_fns=[]).to(device)
    set_cnf_options(margs,model)
    sd = torch.load(args.ckpt,map_location=device)["state_dict"]
    model.load_state_dict(sd); model.eval()

    # ---- metrics -------------------------------------------------------
    kv2 = evaluate_kantorovich_v2(model,data,int_tps,args.leaveout,device)
    mse = evaluate_mse(model,data,int_tps,device)

    # ---- print nice table ---------------------------------------------
    print("\nMetric               | Value")
    print("---------------------|---------------------------")
    print(f"Kantorovich v2       | {kv2}")
    print(f"Kantorovich v2 mean  | {kv2.mean():.4f}")
    print(f"MSE                  | {mse}")
    print(f"MSE mean             | {mse.mean():.4f}")

if __name__ == "__main__":
    main()
