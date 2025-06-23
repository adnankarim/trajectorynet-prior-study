

Minimal repo for training and evaluating TrajectoryNet with different base priors.

## Repository Structure

```bash
git clone https://github.com/adnankarim/trajectorynet-prior-study.git
cd trajectorynet-prior-study
├── TrajectoryNet    #main repo for trian,eval files
├── scripts/         # Training, Visualization and evaluation scripts
├── priors-weights/          #base fitted priors weights
├── results/         # Output metrics and logs
├── requirements.txt # Dependencies
├── README.md        # This file
├── LICENSE.md       # License
```

## Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train TrajectoryNet via module invocation with a specified base prior:

```bash
python -m TrajectoryNet.train \
  --dataset DATASET_NAME \
  --prior PRIOR_NAME \
  --output-dir results/PRIOR_NAME_run
```

Supported priors: `gaussian`, `gmm`, `gmmtorch`, `empirical`, `nsf`

### Evaluation

Evaluate a trained model using the `eval` module, specifying embedding and leave-out timepoint:

```bash
python -m TrajectoryNet.eval \
  --dataset DATASET_NAME \
  --embedding_name PCA \
  --leaveout_timepoint 1 \
  --save results/PRIOR_NAME_run
```

#### Batch Evaluation Example (Windows)

```bat
for %%d in (circle5_base circle5_baseD circle5_baseDV circle5_base_V) do (
  python -m TrajectoryNet.eval --dataset CIRCLE5 --embedding_name PCA --leaveout_timepoint 1 --save results/%%d
)
```

## License

MIT License. See [LICENSE.md](LICENSE.md).

## Citation

> Tong, A., Huang, J., Wolf, G., van Dijk, D., & Krishnaswamy, S. (2020). TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics. *ICML 2020*.
