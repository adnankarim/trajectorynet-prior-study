# run_eb_energy.sh  ← place this next to your TrajectoryNet/ directory

#!/usr/bin/env bash
set -euo pipefail

# Common args
COMMON="--solver dopri5 \
        --niters 10000 \
        --embedding_name phate \
        --save_freq 1000 \
        --viz_freq 1000"

# Energy regularization coefficient
ENERGY_COEFF=0.1

for TP in 1 2 3; do
  echo "=== Running EB Base+E, leaveout_timepoint=$TP ==="
  python -m TrajectoryNet.main \
    $COMMON \
    --dataset EB \
    --leaveout_timepoint $TP \
    --save results/eb_base_E_t${TP} \
    --sl2int $ENERGY_COEFF
done
