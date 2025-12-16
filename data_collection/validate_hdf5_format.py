"""
HDF5 Format Validation Tool

Validates that collected data files meet the specification for
qubit control pulse calibration datasets.

Usage:
    python experiments/hardware/validate_hdf5_format.py <filename.h5>
"""

import sys
import json
from pathlib import Path

import h5py
import numpy as np


def validate_hdf5_format(filepath: str) -> bool:
    """
    Validate HDF5 file against the specification.

    Returns True if valid, False otherwise.
    """

    filepath = Path(filepath)
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return False

    print(f"\n{'='*70}")
    print(f"VALIDATING: {filepath.name}")
    print(f"{'='*70}\n")

    all_checks_passed = True

    with h5py.File(filepath, 'r') as f:

        # ===== CHECK METADATA GROUP =====
        print("[1] Checking /metadata group...")

        if 'metadata' not in f:
            print("  ❌ Missing /metadata group")
            all_checks_passed = False
        else:
            metadata = f['metadata']
            required_attrs = ['qubits', 'sample_rate_ns', 'n_shots', 'timestamp']

            for attr in required_attrs:
                if attr in metadata.attrs:
                    value = metadata.attrs[attr]
                    if attr == 'qubits':
                        qubits = json.loads(value)
                        print(f"  ✓ qubits: {qubits}")
                    else:
                        print(f"  ✓ {attr}: {value}")
                else:
                    print(f"  ❌ Missing attribute: {attr}")
                    all_checks_passed = False

            if 'notes' in metadata.attrs:
                print(f"  ✓ notes: {metadata.attrs['notes'][:50]}...")

        # ===== CHECK PULSE CONFIGS TABLE =====
        print("\n[2] Checking /drive_configs dataset...")

        if 'drive_configs' not in f:
            print("  ❌ Missing /drive_configs dataset")
            all_checks_passed = False
        else:
            configs = f['drive_configs']
            print(f"  ✓ Found drive_configs with shape: {configs.shape}")
            print(f"  ✓ dtype: {configs.dtype}")

            # Check required fields
            required_fields = [
                'drive_qubit_idx', 'pulse_type', 'amplitude', 'duration_ns',
                'detuning_Hz', 'phase_rad'
            ]

            for field in required_fields:
                if field in configs.dtype.names:
                    print(f"  ✓ Field '{field}' present")
                else:
                    print(f"  ❌ Missing field: {field}")
                    all_checks_passed = False

            # Show first few entries
            print(f"\n  First 3 pulse configurations:")
            for i in range(min(3, len(configs))):
                row = configs[i]
                print(f"    [{i}] drive_q={row['drive_qubit_idx']}, "
                      f"amp={row['amplitude']:.4f}, "
                      f"type={row['pulse_type'].decode() if isinstance(row['pulse_type'], bytes) else row['pulse_type']}")

        # ===== CHECK RAW IQ DATA =====
        print("\n[3] Checking /raw_iq group...")

        if 'raw_iq' not in f:
            print("  ❌ Missing /raw_iq group")
            all_checks_passed = False
        else:
            raw_iq = f['raw_iq']

            # Check I and Q datasets
            for component in ['I', 'Q']:
                if component not in raw_iq:
                    print(f"  ❌ Missing /raw_iq/{component} dataset")
                    all_checks_passed = False
                else:
                    data = raw_iq[component]
                    print(f"  ✓ {component} shape: {data.shape}")
                    print(f"    dtype: {data.dtype}")
                    print(f"    size: {data.size * data.dtype.itemsize / 1e6:.2f} MB")

                    # Validate 4D structure
                    if len(data.shape) != 4:
                        print(f"  ❌ Expected 4D array, got {len(data.shape)}D")
                        all_checks_passed = False
                    else:
                        N_pulses, N_qubits, N_shots, N_time = data.shape
                        print(f"    [N_pulses={N_pulses}, N_qubits={N_qubits}, "
                              f"N_shots={N_shots}, N_time={N_time}]")

            # Check attributes
            if 'axis_0' in raw_iq.attrs:
                print(f"\n  Axis labels:")
                for i in range(4):
                    if f'axis_{i}' in raw_iq.attrs:
                        print(f"    axis_{i}: {raw_iq.attrs[f'axis_{i}']}")

        # ===== CROSS-VALIDATION =====
        print("\n[4] Cross-validation checks...")

        if 'drive_configs' in f and 'raw_iq' in f and 'I' in f['raw_iq']:
            n_configs = f['drive_configs'].shape[0]
            n_pulses = f['raw_iq']['I'].shape[0]

            if n_configs == n_pulses:
                print(f"  ✓ Pulse configs count ({n_configs}) matches IQ data ({n_pulses})")
            else:
                print(f"  ❌ Mismatch: {n_configs} configs vs {n_pulses} IQ measurements")
                all_checks_passed = False

        if 'metadata' in f and 'n_shots' in f['metadata'].attrs:
            expected_shots = f['metadata'].attrs['n_shots']
            if 'raw_iq' in f and 'I' in f['raw_iq']:
                actual_shots = f['raw_iq']['I'].shape[2]
                if expected_shots == actual_shots:
                    print(f"  ✓ Shot count matches metadata ({expected_shots})")
                else:
                    print(f"  ⚠️  Warning: metadata says {expected_shots} shots, "
                          f"data has {actual_shots}")

        if 'metadata' in f and 'qubits' in f['metadata'].attrs:
            expected_qubits = len(json.loads(f['metadata'].attrs['qubits']))
            if 'raw_iq' in f and 'I' in f['raw_iq']:
                actual_qubits = f['raw_iq']['I'].shape[1]
                if expected_qubits == actual_qubits:
                    print(f"  ✓ Qubit count matches metadata ({expected_qubits})")
                else:
                    print(f"  ❌ Metadata says {expected_qubits} qubits, "
                          f"data has {actual_qubits}")
                    all_checks_passed = False

        # ===== DATA QUALITY CHECKS =====
        print("\n[5] Data quality checks...")

        if 'raw_iq' in f and 'I' in f['raw_iq'] and 'Q' in f['raw_iq']:
            I_data = f['raw_iq']['I']
            Q_data = f['raw_iq']['Q']

            # Check for NaN/Inf
            if np.any(np.isnan(I_data[:])) or np.any(np.isnan(Q_data[:])):
                print("  ❌ Data contains NaN values")
                all_checks_passed = False
            else:
                print("  ✓ No NaN values")

            if np.any(np.isinf(I_data[:])) or np.any(np.isinf(Q_data[:])):
                print("  ❌ Data contains Inf values")
                all_checks_passed = False
            else:
                print("  ✓ No Inf values")

            # Show data statistics
            print(f"\n  I statistics:")
            print(f"    mean: {np.mean(I_data[:]):.6f}")
            print(f"    std:  {np.std(I_data[:]):.6f}")
            print(f"    min:  {np.min(I_data[:]):.6f}")
            print(f"    max:  {np.max(I_data[:]):.6f}")

            print(f"\n  Q statistics:")
            print(f"    mean: {np.mean(Q_data[:]):.6f}")
            print(f"    std:  {np.std(Q_data[:]):.6f}")
            print(f"    min:  {np.min(Q_data[:]):.6f}")
            print(f"    max:  {np.max(Q_data[:]):.6f}")

    # ===== FINAL RESULT =====
    print(f"\n{'='*70}")
    if all_checks_passed:
        print("✅ VALIDATION PASSED - File meets specification")
    else:
        print("❌ VALIDATION FAILED - See errors above")
    print(f"{'='*70}\n")

    return all_checks_passed


def print_file_structure(filepath: str):
    """Print the complete HDF5 file structure"""

    print(f"\n{'='*70}")
    print("FILE STRUCTURE")
    print(f"{'='*70}\n")

    with h5py.File(filepath, 'r') as f:

        def print_attrs(name, obj):
            """Helper to print attributes"""
            indent = "  " * (name.count('/'))
            if isinstance(obj, h5py.Group):
                print(f"{indent}{name}/ (Group)")
                if obj.attrs:
                    for key, val in obj.attrs.items():
                        print(f"{indent}  @{key} = {val}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}{name} (Dataset) shape={obj.shape}, dtype={obj.dtype}")
                if obj.attrs:
                    for key, val in obj.attrs.items():
                        print(f"{indent}  @{key} = {val}")

        f.visititems(print_attrs)

    print()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python validate_hdf5_format.py <filename.h5>")
        print("\nExample:")
        print("  python experiments/hardware/validate_hdf5_format.py \\")
        print("    data/20251211_143000_qubit_control_calibration.h5")
        sys.exit(1)

    filepath = sys.argv[1]

    # Show structure
    print_file_structure(filepath)

    # Validate
    is_valid = validate_hdf5_format(filepath)

    sys.exit(0 if is_valid else 1)
