"""
HDF5 Dataset Loader for Quantum Pulse Calibration Data

Loads amplitude sweep datasets collected from Qolab hardware
and provides interfaces for ML training pipelines.
"""

import h5py
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path


class TimeResolvedIQDataset:
    """
    Loader for HDF5 amplitude sweep datasets.

    Expected HDF5 structure:
        /metadata (group with attrs)
        /drive_configs (structured array)
        /raw_iq/I (float array [N_pulses, N_qubits, N_shots, N_time])
        /raw_iq/Q (float array [N_pulses, N_qubits, N_shots, N_time])
    """

    def __init__(self, filepath: str):
        """
        Args:
            filepath: Path to HDF5 file
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        with h5py.File(self.filepath, 'r') as f:
            # Load metadata
            self.qubits = eval(f['metadata'].attrs['qubits'])
            self.n_shots = f['metadata'].attrs['n_shots']
            self.timestamp = f['metadata'].attrs['timestamp']

            # Load drive configurations
            self.drive_configs = f['drive_configs'][:]

            # Load IQ data (keep in file, load on demand)
            self.n_pulses, self.n_qubits, _, self.n_time = f['raw_iq/I'].shape

    def get_pulse_parameters(self, fields: Optional[List[str]] = None) -> np.ndarray:
        """
        Get pulse parameter table.

        Args:
            fields: List of field names to extract (None = all fields)

        Returns:
            Array of shape [N_pulses, n_features]
        """
        with h5py.File(self.filepath, 'r') as f:
            configs = f['drive_configs'][:]

        if fields is None:
            # Default: numeric fields only
            fields = ['drive_qubit_idx', 'amplitude_factor', 'duration_ns',
                     'phase_rad', 'detuning_Hz']

        features = []
        for field in fields:
            if field in configs.dtype.names:
                features.append(configs[field])

        return np.column_stack(features)

    def get_iq_responses(self, average_shots: bool = True) -> np.ndarray:
        """
        Get IQ measurement responses.

        Args:
            average_shots: If True, average over shots dimension

        Returns:
            If average_shots=True: [N_pulses, N_qubits, N_time, 2] (last dim: I, Q)
            If average_shots=False: [N_pulses, N_qubits, N_shots, N_time, 2]
        """
        with h5py.File(self.filepath, 'r') as f:
            I = f['raw_iq/I'][:]
            Q = f['raw_iq/Q'][:]

        if average_shots:
            I_avg = np.mean(I, axis=2)  # [N_pulses, N_qubits, N_time]
            Q_avg = np.mean(Q, axis=2)
            return np.stack([I_avg, Q_avg], axis=-1)
        else:
            return np.stack([I, Q], axis=-1)

    def get_training_data(self,
                         target_qubit_idx: int,
                         average_shots: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get (X, y) for supervised learning.

        Args:
            target_qubit_idx: Index of qubit to predict (0-4 for 5-qubit system)
            average_shots: Whether to average shots

        Returns:
            X: Pulse parameters [N_pulses, n_features]
            y: IQ responses for target qubit [N_pulses, N_time, 2] or [N_pulses, N_shots, N_time, 2]
        """
        X = self.get_pulse_parameters()
        iq_all = self.get_iq_responses(average_shots=average_shots)

        if average_shots:
            y = iq_all[:, target_qubit_idx, :, :]  # [N_pulses, N_time, 2]
        else:
            y = iq_all[:, target_qubit_idx, :, :, :]  # [N_pulses, N_shots, N_time, 2]

        return X, y

    def estimate_crosstalk_matrix(self, method: str = 'amplitude_ratio') -> np.ndarray:
        """
        Estimate crosstalk matrix H from one-hot amplitude sweep data.

        Args:
            method: 'amplitude_ratio' or 'rms_ratio'

        Returns:
            H: Crosstalk matrix [N_qubits, N_qubits] where H[i,j] =
               signal at qubit i when driving qubit j (normalized to H[j,j]=1)
        """
        iq = self.get_iq_responses(average_shots=True)  # [N_pulses, N_qubits, N_time, 2]
        configs = self.get_pulse_parameters(['drive_qubit_idx'])

        H = np.zeros((self.n_qubits, self.n_qubits))

        for drive_idx in range(self.n_qubits):
            # Find pulses where this qubit was driven
            mask = configs[:, 0] == drive_idx
            iq_subset = iq[mask]  # [N_amp_steps, N_qubits, N_time, 2]

            if method == 'amplitude_ratio':
                # Use max IQ magnitude across amplitude sweep
                magnitudes = np.sqrt(iq_subset[..., 0]**2 + iq_subset[..., 1]**2)
                max_response = np.max(magnitudes, axis=(0, 2))  # [N_qubits]
            elif method == 'rms_ratio':
                # Use RMS across amplitude sweep and time
                magnitudes = np.sqrt(iq_subset[..., 0]**2 + iq_subset[..., 1]**2)
                max_response = np.sqrt(np.mean(magnitudes**2, axis=(0, 2)))
            else:
                raise ValueError(f"Unknown method: {method}")

            # Normalize to diagonal element
            H[:, drive_idx] = max_response / (max_response[drive_idx] + 1e-12)

        return H

    def __repr__(self):
        return (f"TimeResolvedIQDataset(qubits={self.qubits}, "
                f"n_pulses={self.n_pulses}, n_shots={self.n_shots}, "
                f"timestamp={self.timestamp})")


if __name__ == "__main__":
    # Example usage
    dataset = TimeResolvedIQDataset("../data/20251211_221504_amplitude_sweep_dataset.h5")
    print(dataset)

    # Get training data for qubit 0
    X, y = dataset.get_training_data(target_qubit_idx=0)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Estimate crosstalk matrix
    H = dataset.estimate_crosstalk_matrix()
    print("Crosstalk matrix H:")
    print(H)
    print(f"Diagonal (should be ~1.0): {np.diag(H)}")
