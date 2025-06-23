import argparse
from .lib.layers import odefunc

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams"]

parser = argparse.ArgumentParser("Continuous Normalizing Flow")
parser.add_argument("--test", action="store_true")
parser.add_argument("--dataset", type=str, default="EB")
parser.add_argument("--use_growth", action="store_true")
parser.add_argument("--use_density", action="store_true")
parser.add_argument("--leaveout_timepoint", type=int, default=-1)
parser.add_argument(
    "--layer_type",
    type=str,
    default="concatsquash",
    choices=[
        "ignore",
        "concat",
        "concat_v2",
        "squash",
        "concatsquash",
        "concatcoord",
        "hyper",
        "blend",
    ],
)
parser.add_argument("--max_dim", type=int, default=10)
parser.add_argument("--dims", type=str, default="64-64-64")
parser.add_argument("--num_blocks", type=int, default=1, help="Number of stacked CNFs.")
parser.add_argument("--time_scale", type=float, default=0.5)
parser.add_argument("--train_T", type=eval, default=True)
parser.add_argument(
    "--divergence_fn",
    type=str,
    default="brute_force",
    choices=["brute_force", "approximate"],
)
parser.add_argument(
    "--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES
)
parser.add_argument("--stochastic", action="store_true")

parser.add_argument(
    "--alpha", type=float, default=0.0, help="loss weight parameter for growth model"
)
parser.add_argument("--solver", type=str, default="dopri5", choices=SOLVERS)
parser.add_argument("--atol", type=float, default=1e-5)
parser.add_argument("--rtol", type=float, default=1e-5)
parser.add_argument(
    "--step_size", type=float, default=None, help="Optional fixed step size."
)

parser.add_argument("--test_solver", type=str, default=None, choices=SOLVERS + [None])
parser.add_argument("--test_atol", type=float, default=None)
parser.add_argument("--test_rtol", type=float, default=None)

parser.add_argument("--residual", action="store_true")
parser.add_argument("--rademacher", action="store_true")
parser.add_argument("--spectral_norm", action="store_true")
parser.add_argument("--batch_norm", action="store_true")
parser.add_argument("--bn_lag", type=float, default=0)

parser.add_argument("--niters", type=int, default=10000)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--test_batch_size", type=int, default=1000)
parser.add_argument("--viz_batch_size", type=int, default=2000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-5)

# Track quantities
parser.add_argument("--l1int", type=float, default=None, help="int_t ||f||_1")
parser.add_argument("--l2int", type=float, default=None, help="int_t ||f||_2")
parser.add_argument("--sl2int", type=float, default=None, help="int_t ||f||_2^2")
parser.add_argument(
    "--dl2int", type=float, default=None, help="int_t ||f^T df/dt||_2"
)  # f df/dx?
parser.add_argument(
    "--dtl2int", type=float, default=None, help="int_t ||f^T df/dx + df/dt||_2"
)
parser.add_argument("--JFrobint", type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument(
    "--JdiagFrobint", type=float, default=None, help="int_t ||df_i/dx_i||_F"
)
parser.add_argument(
    "--JoffdiagFrobint", type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F"
)
parser.add_argument("--vecint", type=float, default=None, help="regularize direction")
parser.add_argument(
    "--use_magnitude",
    action="store_true",
    help="regularize direction using MSE loss instead of cosine loss",
)

parser.add_argument(
    "--interp_reg", type=float, default=None, help="regularize interpolation"
)

parser.add_argument("--save", type=str, default="../results/tmp")
parser.add_argument("--save_freq", type=int, default=1000)
parser.add_argument("--viz_freq", type=int, default=100)
parser.add_argument("--viz_freq_growth", type=int, default=100)
parser.add_argument("--val_freq", type=int, default=100)
parser.add_argument("--log_freq", type=int, default=10)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--no_display_loss", action="store_false")
parser.add_argument(
    "--top_k_reg", type=float, default=0.0, help="density following regularization"
)
parser.add_argument("--training_noise", type=float, default=0.1)
parser.add_argument(
    "--embedding_name",
    type=str,
    default="pca",
    help="choose embedding name to perform TrajectoryNet on",
)
parser.add_argument("--whiten", action="store_true", help="Whiten data before running TrajectoryNet")
parser.add_argument("--save_movie", action="store_false", help="Construct trajectory movie, requires ffmpeg to be installed")
parser.add_argument(
    "--prior",
    default="gmmt",           # choices: gaussian | gmm | nsf …
    choices=["gaussian", "gmm","gmmt", "nsf","empirical"],
    help="Type of base‐density used for p(z₀)"
)
# optional: where to store / load a fitted GMM
parser.add_argument("--gmm_path", default="priors/{dataset}_t1_gmm.pkl")
parser.add_argument("--nsf_path", default="priors/{dataset}_nsf.pt")

parser.add_argument("--seeds", type=int, default=3,
                    help="number of random seeds to average (default 3)")
parser.add_argument("--csv",   default="bio-eval.csv",
                    help="output CSV filename (default bio-eval.csv)")
# python -m TrajectoryNet.main 
#        --dataset TREE 
#        --save results/tree_base_V 
#        --solver dopri5 
#        --niters 10000 

# python -m TrajectoryNet.eval --dataset EB --embedding_name phate --solver dopri5 --leaveout_timepoint 1 --save results/eb_base_t1
# for tp in 1 2 3; do
#   python -m TrajectoryNet.main \
#     $COMMON \
#     --leaveout_timepoint $tp \
#     --l2int 1e-2 \
#     --save results/eb_E_t$tp
# done
# for tp in 3; do
#   python -m TrajectoryNet.main \
#     $COMMON \
#     --dataset EB \
#     --leaveout_timepoint $tp \
#     --save bio/eb_D_t$tp
# done


# for d in arch_base arch_baseLINE circle5_base circle5_baseD circle5_baseDV circle5_base_V cycle_base cycle_baseD cycle_baseDV cycle_base_V tree_base tree_baseD tree_baseDV tree_base_V; do \
#   python -m TrajectoryNet.eval \
#     --dataset ~/Desktop/project/TrajectoryNet-master/results/$d/data.npz \
#     --embedding_name PCA \
#     --leave_out_idx 2 \
#     --save ~/Desktop/project/TrajectoryNet-master/results/$d; \
# done

# for tp in 3; do
#   python -m TrajectoryNet.main \
#     $COMMON \
#     --dataset EB \
#     --leaveout_timepoint $tp \
#     --save bio/eb_DV_t$tp
# done

# for tp in 1 2 3; do
#   python -m TrajectoryNet.main \
#     $COMMON \
#     --dataset EB \
#     --leaveout_timepoint $tp \
#     --save bio/eb_E_t$tp \
#     --sl2int 0.1
# done


# python -m TrajectoryNet.main 
#        --dataset TREEPAPER 
#        --save res/tree_base 
#        --solver dopri5 
#        --niters 10000 


# python -m TrajectoryNet.main --dataset CYCLE --save experiments/cycle_b_delta_gmm10000 --solver dopri5 --niters 10000 --lr 1e-4 --spectral_norm --l2int 1e-2 --whiten --training_noise 0.02 --time_scale 1