@REM :: Evaluation for EB datasets at timepoint t1
@REM for %%d in (eb_base_t1 eb_D_t1 eb_V_t1 eb_DV_t1) do (
@REM     python -m TrajectoryNet.eval --dataset EB --embedding_name PCA --leaveout_timepoint 1 --save bio/%%d
@REM )

@REM :: Evaluation for EB datasets at timepoint t2
@REM for %%d in (eb_base_t2 eb_D_t2 eb_V_t2 eb_DV_t2) do (
@REM     python -m TrajectoryNet.eval --dataset EB --embedding_name PCA --leaveout_timepoint 2 --save bio/%%d
@REM )

@REM :: Evaluation for EB datasets at timepoint t3
@REM for %%d in (eb_base_t3 eb_D_t3 eb_V_t3 eb_DV_t3) do (
@REM     python -m TrajectoryNet.eval --dataset EB --embedding_name PCA --leaveout_timepoint 3 --save bio/%%d
@REM )
:: Evaluation for EB datasets at timepoint t3
@REM for %%d in (eb_base_E_t1) do (
@REM     python -m TrajectoryNet.eval --dataset EB --embedding_name PCA --leaveout_timepoint 1 --save energy/%%d
@REM )
@REM for %%d in (eb_base_E_t2 ) do (
@REM     python -m TrajectoryNet.eval --dataset EB --embedding_name PCA --leaveout_timepoint 2 --save energy/%%d
@REM )
for %%d in (eb_base_E_t3) do (
    python -m TrajectoryNet.eval --dataset EB --embedding_name PCA --leaveout_timepoint 3 --save energy/%%d
)
