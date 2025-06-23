
for %%d in (circle5_base circle5_baseD circle5_baseDV circle5_base_V) do (
    python -m TrajectoryNet.eval --dataset CIRCLE5 --embedding_name PCA --leaveout_timepoint 1 --save results/%%d
)
for %%d in (cycle_base cycle_baseD cycle_baseDV cycle_base_V) do (
    python -m TrajectoryNet.eval --dataset CYCLE --embedding_name PCA --leaveout_timepoint 1 --save results/%%d
)
for %%d in (tree_base tree_baseD tree_baseDV tree_base_V) do python -m TrajectoryNet.eval --dataset TREE --embedding_name PCA --leaveout_timepoint 1 --save results/%%d
python -m TrajectoryNet.eval --dataset TREE --embedding_name PCA --leaveout_timepoint 1 --save GMMT/treet