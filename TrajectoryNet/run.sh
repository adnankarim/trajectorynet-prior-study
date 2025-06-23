#!/usr/bin/env bash

# Common arguments for all runs
COMMON="--solver dopri5 \
        --niters 10000 \
        --embedding_name phate \
        --save_freq 1000 \
        --viz_freq 1000"

# Energy regularization coefficient (L_energy)
ENERGY_COEFF=0.1

for TP in -1; do
  python -m TrajectoryNet.main \
    $COMMON \
    --dataset EB \
    --leaveout_timepoint $TP \
    --save results/eb_base_E_t${TP} \
    --sl2int $ENERGY_COEFF
done
