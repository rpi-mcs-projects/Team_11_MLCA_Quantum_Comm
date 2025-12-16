# Machine Learning for Quantum Crosstalk Calibration (MLCA)

**Learning-Based RF Channel Characterization with Crosstalk Modeling for Optimal Transmon Qubit Control**

This repository contains code and datasets for training machine learning models to optimize RF pulse calibration in superconducting quantum computers under crosstalk constraints.

## Quick Start

### Running a Single Backtest

```bash
uv run experiments/closed_loop_backtest.py \
  --npz ./data/amp_sweep_bandpower_bins2_40.npz \
  --budget 20 \
  --seed-points 5 \
  --lambda-xtalk 0.5 \
  --norm-mode oracle \
  --explore-first-k-steps 2 \
  --refine-last-k-steps 2 \
  --proposals 5 \
  --eval-grid 2001 \
  --global-scan-grid 401 \
  --device cpu \
  --seed 0
```

**Parameters:**
- `--budget 20`: Total number of hardware queries allowed
- `--seed-points 5`: Initial random samples before online learning
- `--lambda-xtalk 0.5`: Weight for crosstalk penalty term
- `--explore-first-k-steps 2`: Use exploration (Thompson sampling) for first K steps
- `--refine-last-k-steps 2`: Use exploitation (UCB acquisition) for final K steps
- `--proposals 5`: Number of candidate points generated per iteration
- `--eval-grid 2001`: Grid resolution for ground-truth evaluation
- `--global-scan-grid 401`: Global scan grid resolution
- `--device cpu`: Use CPU (change to `cuda` if GPU available)
- `--seed 0`: Random seed for reproducibility

**Output:** The script runs online training (no separate training phase) and reports final calibration performance.

### Running Monte Carlo Evaluation

```bash
uv run experiments/mc_closed_loop_eval.py
```

This runs multiple backtest trials with different random seeds and aggregates results into `results/monte_carlo_results.csv`. 

**Training occurs online during each backtest** - there is no explicit training command. The training loop is embedded in `experiments/closed_loop_backtest.py`.

### Visualizing Results

```bash
uv run experiments/plot_mc_results.py
```

---

## Repository Structure

```
mlca/
├── README.md                                          # This file
├── requirements.txt                                   # Python dependencies
├── data/                                              # Datasets
│   ├── amp_sweep_bandpower_bins2_40.npz              # Preprocessed (19 KB)
│   └── 20251211_221504_amplitude_sweep_dataset.h5    # Raw HDF5 (3.8 MB)
├── experiments/                                       # Runnable scripts
│   ├── closed_loop_backtest.py                       # Main online learning script
│   ├── mc_closed_loop_eval.py                        # Monte Carlo evaluation
│   └── plot_mc_results.py                            # Results visualization
├── figures/                                           # Publication figures
│   └── 20251211_221504_amplitude_sweep.png           # IQ response matrix
├── results/                                           # Experiment outputs
│   └── monte_carlo_results.csv                       # Precomputed MC statistics
├── src/                                               # Source code
│   ├── __init__.py
│   └── data_loader.py                                # HDF5 dataset loader
├── data_collection/                                   # Dataset collection scripts
│   ├── collect_amplitude_sweep_dataset.py            # Hardware collection script
│   └── validate_hdf5_format.py                       # Dataset validator
└── utils/                                             # Utility modules
    └── data_loader.py                                # HDF5 loading utilities
```

---

## Dataset Description

### Amplitude Sweep Dataset (HDF5)

**File:** `data/20251211_221504_amplitude_sweep_dataset.h5`

Collected on Qolab's 5-qubit superconducting quantum computer using one-hot amplitude sweeps:
- **Platform:** Quantum Machines OPX controller + QuAM state management
- **Protocol:** Drive each qubit individually through 25 amplitude steps (0.5× to 1.5× nominal)
- **Measurements:** 1,000 shots per configuration, simultaneous readout of all 5 qubits
- **Total size:** 125 pulse configs × 5 qubits × 1,000 shots = 625,000 IQ measurements

**HDF5 Structure:**
```
/metadata (group)
  ├── qubits: ['q1', 'q2', 'q3', 'q4', 'q5']
  ├── sample_rate_ns: 4.0
  ├── n_shots: 1000
  └── timestamp: 20251211_221504

/drive_configs (structured array, 125 entries)
  ├── drive_qubit_idx: int (0-4)
  ├── pulse_type: str ('x180')
  ├── amplitude_factor: float (0.5 to 1.5)
  ├── duration_ns: float
  ├── phase_rad: float
  └── detuning_Hz: float

/raw_iq (group)
  ├── I: float32[125, 5, 1000, 1]  # In-phase component
  └── Q: float32[125, 5, 1000, 1]  # Quadrature component
```

**Dimensions:** `[N_pulses, N_qubits, N_shots, N_time]` where `N_time=1` for integrated IQ values.

**Usage Example:**
```python
from src.data_loader import TimeResolvedIQDataset

dataset = TimeResolvedIQDataset('data/20251211_221504_amplitude_sweep_dataset.h5')
pulse_params = dataset.get_pulse_parameters()  # Shape: [125, n_features]
iq_responses = dataset.get_iq_responses()      # Shape: [125, 5, 1000, 1, 2]
```

### Preprocessed Data (NPZ)

**File:** `data/amp_sweep_bandpower_bins2_40.npz`

Preprocessed version of the HDF5 dataset with bandpower features extracted from IQ traces. Used directly by `experiments/closed_loop_backtest.py`.

---

## Hardware Platform

**Qolab 5-Qubit Superconducting Processor:**
- **Qubits:** q1, q2, q3, q4, q5 (transmon architecture)
- **Coherence times:**
  - T₁: 55-129 μs
  - T₂: 2.7-216 μs (q5 has anomalously short T₂)
- **Readout:** Dispersive measurement with 4 μs integration window
- **Control:** IQ-modulated microwave pulses via Quantum Machines OPX
- **Access:** Cloud-based via IQCC platform

**Observed Crosstalk:** Mean isolation ~24.6 dB between qubit pairs (see `figures/20251211_221504_amplitude_sweep.png` for off-diagonal crosstalk visualization).

---

## Data Collection

To replicate the dataset collection on Qolab hardware, see `data_collection/collect_amplitude_sweep_dataset.py`. This script demonstrates the hardware protocol and data format.

**Note:** Data collection requires hardware access and cloud credentials. The provided datasets are sufficient for running all experiments without hardware access.

---

## Dependencies

**Core:**
- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy

**Data Processing:**
- h5py (HDF5 file I/O)
- pandas (MC results)
- tqdm (progress bars)

**Install:**
```bash
pip install -r requirements.txt
```

---

## Training

Training occurs **online during the backtest execution**. The `experiments/closed_loop_backtest.py` script implements an adaptive online learning algorithm that:

1. Starts with random seed points (`--seed-points`)
2. Trains a surrogate model iteratively
3. Uses acquisition functions (Thompson sampling or UCB) to propose new points
4. Runs up to `--budget` hardware queries total

No separate training phase is needed. Training and evaluation are integrated.

---

## Citation

If you use this dataset or code, please cite:

```bibtex
@misc{mlca2024,
  title={Learning-Based RF Channel Characterization with Crosstalk Modeling for Optimal Transmon Qubit Control},
  author={Zhang, Hisen and Fiumara, Dan},
  year={2024},
  howpublished={\url{https://github.com/rpi-mcs-projects/MLCA}},
  note={Dataset collected on Qolab 5-qubit processor, December 2024}
}
```

---

## License

MIT 2025

---

## Contact

For questions about the dataset or code:
- Dataset collection: Hisen Zhang (zhangz29@rpi.edu)
- ML training algorithms: Dan Fiumara (fiumad@rpi.edu)

**Acknowledgments:** Data collected using Qolab's cloud quantum computing platform.
