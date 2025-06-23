# trajectorynet-prior-study

# trajectorynet-reproduction

A reproduction of the synthetic experiments from **TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics** by Tong *et al.* (ICML 2020).

## Repository Structure

```
trajectorynet-reproduction/
├── data/                  # Synthetic datasets (TREE, ARCH, CYCLE)
├── notebooks/             # Jupyter notebooks for data preprocessing, training, and evaluation
├── src/                   # Python scripts for model implementation and utilities
│   ├── data_utils.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── results/               # Generated figures, metrics (EMD, MSE)
├── environment.yml        # Conda environment specification
└── README.md              # Project overview and instructions
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/adnankarim/trajectorynet-prior-study
   cd trajectorynet-reproduction
   ```

2. Create and activate the conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate trajnet
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preprocessing:**

   ```bash
   python src/data_utils.py --dataset ARCH --preprocess
   ```

2. **Training:**

   ```bash
   python src/train.py --dataset TREE --base gaussian --epochs 10000
   ```

3. **Evaluation:**

   ```bash
   python src/evaluate.py --dataset CYCLE --metrics emd mse
   ```

4. **Notebooks:**
   Open `notebooks/analysis.ipynb` for visualizations and detailed walkthroughs.

## Results

* Quantitative metrics (EMD, MSE) for ARCH, TREE, and CYCLE datasets.
* Comparison of base distributions: Gaussian, GMM, Neural Spline Flow.
* Figures illustrating interpolated trajectories.

See the `results/` directory for detailed plots and tables.

## License

MIT License. Feel free to use and modify.

## Citation

If you use this code, please cite:

> Tong, A., Huang, J., Wolf, G., van Dijk, D., & Krishnaswamy, S. (2020). TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics. *ICML 2020*.
