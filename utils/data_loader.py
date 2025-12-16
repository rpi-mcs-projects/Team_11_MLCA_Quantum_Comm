"""
Time-Resolved IQ Data Loader

Loads and processes HDF5 datasets for ML-based qubit control optimization.

Usage:
    from src.calibration.time_resolved_iq_loader import TimeResolvedIQDataset

    dataset = TimeResolvedIQDataset('data/20251211_143000_qubit_control_calibration.h5')

    # Get pulse parameters and IQ responses
    pulse_params = dataset.get_pulse_parameters()  # [N_pulses, feature_dim]
    iq_responses = dataset.get_iq_responses()      # [N_pulses, N_qubits, N_shots, N_time, 2]

    # Or get preprocessed data for training
    X, y = dataset.get_training_data(target_qubit_idx=0)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


class TimeResolvedIQDataset:
    """
    Dataset loader for time-resolved IQ calibration data.

    Provides convenient access to pulse parameters and IQ responses,
    with options for preprocessing and normalization.
    """

    def __init__(
        self,
        filepath: str,
        normalize: bool = True,
        average_shots: bool = False
    ):
        """
        Load HDF5 dataset.

        Args:
            filepath: Path to HDF5 file
            normalize: Whether to normalize IQ values to zero mean, unit variance
            average_shots: Whether to average over shots dimension
        """
        self.filepath = Path(filepath)
        self.normalize = normalize
        self.average_shots = average_shots

        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Load metadata
        with h5py.File(self.filepath, 'r') as f:
            self.qubits = json.loads(f['metadata'].attrs['qubits'])
            self.n_qubits = len(self.qubits)
            self.n_shots = f['metadata'].attrs['n_shots']
            self.sample_rate_ns = f['metadata'].attrs['sample_rate_ns']
            self.timestamp = f['metadata'].attrs['timestamp']

            # Get dimensions
            self.n_pulses = f['drive_configs'].shape[0]
            self.n_time = f['raw_iq']['I'].shape[3]

        # Normalization statistics (computed on first access)
        self._I_mean = None
        self._I_std = None
        self._Q_mean = None
        self._Q_std = None

    def __repr__(self) -> str:
        return (
            f"TimeResolvedIQDataset(\n"
            f"  file='{self.filepath.name}',\n"
            f"  n_pulses={self.n_pulses},\n"
            f"  n_qubits={self.n_qubits},\n"
            f"  n_shots={self.n_shots},\n"
            f"  n_time={self.n_time},\n"
            f"  qubits={self.qubits}\n"
            f")"
        )

    def get_metadata(self) -> Dict:
        """Get all metadata as dictionary"""
        with h5py.File(self.filepath, 'r') as f:
            metadata = dict(f['metadata'].attrs)
            # Decode bytes to strings
            metadata['qubits'] = json.loads(metadata['qubits'])
            if 'notes' in metadata:
                if isinstance(metadata['notes'], bytes):
                    metadata['notes'] = metadata['notes'].decode()
        return metadata

    def get_pulse_parameters(
        self,
        fields: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Get pulse parameters as feature matrix.

        Args:
            fields: List of field names to include. If None, uses:
                ['drive_qubit_idx', 'amplitude', 'duration_ns', 'detuning_Hz',
                 'phase_rad', 'drag_alpha', 'sigma_ns']

        Returns:
            Array of shape [N_pulses, n_features]
        """
        if fields is None:
            fields = [
                'drive_qubit_idx',
                'amplitude',
                'duration_ns',
                'detuning_Hz',
                'phase_rad',
                'drag_alpha',
                'sigma_ns'
            ]

        with h5py.File(self.filepath, 'r') as f:
            configs = f['drive_configs'][:]

            # Extract and stack fields
            features = []
            for field in fields:
                if field in configs.dtype.names:
                    features.append(configs[field])
                else:
                    raise ValueError(f"Field '{field}' not found in drive_configs")

            # Stack into matrix
            X = np.column_stack(features)

        return X

    def get_iq_responses(
        self,
        apply_normalization: bool = None,
        apply_averaging: bool = None
    ) -> np.ndarray:
        """
        Get IQ response data.

        Args:
            apply_normalization: Override instance normalize setting
            apply_averaging: Override instance average_shots setting

        Returns:
            If average_shots=False:
                Array of shape [N_pulses, N_qubits, N_shots, N_time, 2]
                where last dimension is [I, Q]
            If average_shots=True:
                Array of shape [N_pulses, N_qubits, N_time, 2]
        """
        if apply_normalization is None:
            apply_normalization = self.normalize
        if apply_averaging is None:
            apply_averaging = self.average_shots

        with h5py.File(self.filepath, 'r') as f:
            I = f['raw_iq']['I'][:]
            Q = f['raw_iq']['Q'][:]

        # Compute normalization statistics if needed
        if apply_normalization and self._I_mean is None:
            self._I_mean = np.mean(I)
            self._I_std = np.std(I)
            self._Q_mean = np.mean(Q)
            self._Q_std = np.std(Q)

        # Apply normalization
        if apply_normalization:
            I = (I - self._I_mean) / (self._I_std + 1e-8)
            Q = (Q - self._Q_mean) / (self._Q_std + 1e-8)

        # Average over shots if requested
        if apply_averaging:
            I = np.mean(I, axis=2)  # [N_pulses, N_qubits, N_time]
            Q = np.mean(Q, axis=2)

        # Stack I and Q
        if apply_averaging:
            # Shape: [N_pulses, N_qubits, N_time, 2]
            iq = np.stack([I, Q], axis=-1)
        else:
            # Shape: [N_pulses, N_qubits, N_shots, N_time, 2]
            iq = np.stack([I, Q], axis=-1)

        return iq

    def get_training_data(
        self,
        target_qubit_idx: int,
        include_all_qubit_responses: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data formatted for ML training.

        Args:
            target_qubit_idx: Which qubit's responses to predict
            include_all_qubit_responses: Whether to include all qubits in output
                (True for crosstalk modeling, False for single-qubit)

        Returns:
            X: Pulse parameters [N_pulses, n_features]
            y: IQ responses
                If include_all_qubit_responses=True and average_shots=True:
                    [N_pulses, N_qubits, N_time, 2]
                If include_all_qubit_responses=False and average_shots=True:
                    [N_pulses, N_time, 2]
                If shots not averaged, adds shot dimension
        """
        X = self.get_pulse_parameters()
        iq = self.get_iq_responses()

        if not include_all_qubit_responses:
            if self.average_shots:
                # Select target qubit: [N_pulses, N_time, 2]
                y = iq[:, target_qubit_idx, :, :]
            else:
                # Select target qubit: [N_pulses, N_shots, N_time, 2]
                y = iq[:, target_qubit_idx, :, :, :]
        else:
            y = iq

        return X, y

    def get_crosstalk_matrix_estimates(
        self,
        use_amplitude: bool = True
    ) -> np.ndarray:
        """
        Estimate crosstalk matrix from IQ response amplitudes.

        For each drive qubit, measure the response amplitude on all qubits
        to estimate crosstalk coefficients.

        Args:
            use_amplitude: Use IQ amplitude (sqrt(I^2 + Q^2)), else use I component

        Returns:
            Estimated crosstalk matrix [N_qubits, N_qubits]
            where H[i,j] = response at qubit i when driving qubit j
        """
        iq = self.get_iq_responses(apply_averaging=True)  # [N_pulses, N_qubits, N_time, 2]

        H = np.zeros((self.n_qubits, self.n_qubits))

        # Get pulse parameters to identify which qubit was driven
        params = self.get_pulse_parameters(fields=['drive_qubit_idx', 'amplitude'])

        for drive_q_idx in range(self.n_qubits):
            # Find pulses that drove this qubit
            mask = params[:, 0] == drive_q_idx

            # Get IQ responses for these pulses
            iq_subset = iq[mask]  # [N_amps, N_qubits, N_time, 2]

            # Compute response amplitude for each measured qubit
            for meas_q_idx in range(self.n_qubits):
                I = iq_subset[:, meas_q_idx, :, 0]
                Q = iq_subset[:, meas_q_idx, :, 1]

                if use_amplitude:
                    amplitude = np.sqrt(I**2 + Q**2)
                else:
                    amplitude = np.abs(I)

                # Take peak response across time and amplitude sweep
                H[meas_q_idx, drive_q_idx] = np.max(amplitude)

        # Normalize so diagonal is 1
        for i in range(self.n_qubits):
            if H[i, i] > 0:
                H[:, i] /= H[i, i]

        return H

    def filter_by_drive_qubit(
        self,
        drive_qubit_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for a specific drive qubit only.

        Args:
            drive_qubit_idx: Index of drive qubit

        Returns:
            X: Pulse parameters for this drive qubit
            y: IQ responses [N_pulses_subset, N_qubits, ...]
        """
        params = self.get_pulse_parameters(fields=['drive_qubit_idx'])
        mask = params[:, 0] == drive_qubit_idx

        X_full = self.get_pulse_parameters()
        y_full = self.get_iq_responses()

        return X_full[mask], y_full[mask]

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of the dataset"""
        iq = self.get_iq_responses(apply_normalization=False, apply_averaging=False)

        stats = {
            'n_pulses': self.n_pulses,
            'n_qubits': self.n_qubits,
            'n_shots': self.n_shots,
            'n_time': self.n_time,
            'qubits': self.qubits,
            'I_mean': float(np.mean(iq[..., 0])),
            'I_std': float(np.std(iq[..., 0])),
            'I_min': float(np.min(iq[..., 0])),
            'I_max': float(np.max(iq[..., 0])),
            'Q_mean': float(np.mean(iq[..., 1])),
            'Q_std': float(np.std(iq[..., 1])),
            'Q_min': float(np.min(iq[..., 1])),
            'Q_max': float(np.max(iq[..., 1])),
            'amplitude_mean': float(np.mean(np.sqrt(iq[..., 0]**2 + iq[..., 1]**2))),
            'amplitude_std': float(np.std(np.sqrt(iq[..., 0]**2 + iq[..., 1]**2))),
        }

        return stats


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python time_resolved_iq_loader.py <filename.h5>")
        sys.exit(1)

    filepath = sys.argv[1]

    # Load dataset
    print(f"Loading: {filepath}\n")
    dataset = TimeResolvedIQDataset(filepath, normalize=True, average_shots=True)

    print(dataset)
    print()

    # Show metadata
    metadata = dataset.get_metadata()
    print("Metadata:")
    for key, val in metadata.items():
        print(f"  {key}: {val}")
    print()

    # Get training data
    X, y = dataset.get_training_data(target_qubit_idx=0)
    print(f"Training data shapes:")
    print(f"  X (pulse params): {X.shape}")
    print(f"  y (IQ responses): {y.shape}")
    print()

    # Estimate crosstalk matrix
    H_est = dataset.get_crosstalk_matrix_estimates()
    print("Estimated crosstalk matrix:")
    print(H_est)
    print()

    # Summary statistics
    stats = dataset.get_summary_statistics()
    print("Summary statistics:")
    for key, val in stats.items():
        if isinstance(val, (int, float)):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")
