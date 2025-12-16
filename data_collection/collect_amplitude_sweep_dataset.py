"""
AMPLITUDE SWEEP DATA COLLECTION - ONE-HOT ENCODING

Sweeps pulse amplitude for each qubit (one-hot: only one qubit driven at a time)
while measuring all qubits simultaneously to capture crosstalk.

Output:
- HDF5 file with integrated IQ measurements
- Format: [N_pulses, N_qubits, N_shots, N_time] where N_time=1

Each pulse configuration:
- Drive one qubit at one amplitude
- Measure all 5 qubits
- Collect 50K shots for statistical robustness

Total measurements: 5 qubits × N_amp_steps × 50K shots

Usage:
    python experiments/hardware/collect_amplitude_sweep_dataset.py
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit

# Add qolab-start to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "qolab-start"))

from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration


# %% Configuration
class Config:
    """Data collection configuration"""

    # Target qubits (None = all active qubits)
    qubits: Optional[List[str]] = None

    # Amplitude sweep parameters
    min_amp_factor: float = 0.5      # Minimum scaling (0.5× nominal)
    max_amp_factor: float = 1.5      # Maximum scaling (1.5× nominal)
    n_amp_steps: int = 25            # Number of amplitude points

    # Shots per amplitude point
    num_averages: int = 1000

    # Pulse configuration
    drive_operation: str = "x180"    # Pulse type to sweep

    # Measurement settings
    reset_type: Literal["thermal", "active"] = "thermal"
    flux_point: Literal["joint", "independent"] = "independent"
    readout_operation: str = "readout"

    # Execution
    simulate: bool = False
    simulation_duration_ns: int = 10000
    timeout: int = 36000  # 10 hours in seconds

    # Output
    save_dir: Path = Path(__file__).parent.parent.parent / "data"


config = Config()

# %% Initialize QuAM and QOP
print("=" * 80)
print("AMPLITUDE SWEEP DATA COLLECTION - ONE-HOT ENCODING")
print("=" * 80)

u = unit(coerce_to_integer=True)

print("\n[1/6] Loading QuAM state...")
machine = QuAM.load()

# Get qubits
if config.qubits is None:
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in config.qubits]

num_qubits = len(qubits)
qubit_names = [q.name for q in qubits]
print(f"  Using {num_qubits} qubits: {qubit_names}")

# Generate configuration
qua_config = machine.generate_config()

# Connect to QOP
if not config.simulate:
    print("\n[2/6] Connecting to Qolab quantum computer...")
    qmm = machine.connect()
    print("  Connected successfully")
else:
    print("\n[2/6] Simulation mode enabled")

# %% Build amplitude sweep
print("\n[3/6] Building amplitude sweep...")

# Amplitude factors to sweep
amp_factors = np.linspace(
    config.min_amp_factor,
    config.max_amp_factor,
    config.n_amp_steps
)

# Build pulse configuration list
# Each config: (drive_qubit_idx, amplitude_factor)
pulse_configs = []
for drive_q_idx, qubit in enumerate(qubits):
    for amp_factor in amp_factors:
        pulse_configs.append({
            'drive_qubit_idx': drive_q_idx,
            'drive_qubit_name': qubit.name,
            'amplitude_factor': amp_factor,
            'amplitude': qubit.xy.operations[config.drive_operation].amplitude * amp_factor,
            'duration_ns': qubit.xy.operations[config.drive_operation].length,
            'drag_alpha': getattr(qubit.xy.operations[config.drive_operation], 'drag_alpha', 0.0),
            'sigma_ns': getattr(qubit.xy.operations[config.drive_operation], 'sigma',
                               qubit.xy.operations[config.drive_operation].length / 5.0),
        })

n_pulses = len(pulse_configs)

print(f"  Total pulse configurations: {n_pulses}")
print(f"    = {num_qubits} qubits × {config.n_amp_steps} amplitude steps")
print(f"  Amplitude range: {config.min_amp_factor:.2f}× to {config.max_amp_factor:.2f}× nominal")
print(f"  Shots per config: {config.num_averages}")
print(f"  Total measurements: {n_pulses * config.num_averages}")

# %% QUA Program
print("\n[4/6] Compiling QUA program...")

with program() as amplitude_sweep:
    # Declare variables
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)

    # Loop variables
    qubit_idx = declare(int)
    amp_idx = declare(int)
    shot = declare(int)
    n_count = declare(int)

    # Amplitude scaling factor
    amp_scale = declare(fixed)

    # Set flux points
    for qubit in qubits:
        machine.set_all_fluxes(flux_point=config.flux_point, target=qubit)

    # Outer loop: drive qubit
    with for_(qubit_idx, 0, qubit_idx < num_qubits, qubit_idx + 1):

        # Middle loop: amplitude
        with for_(*from_array(amp_scale, amp_factors)):

            # Inner loop: shots
            with for_(shot, 0, shot < config.num_averages, shot + 1):

                # Progress tracking
                assign(n_count,
                       qubit_idx * config.n_amp_steps * config.num_averages +
                       shot * config.n_amp_steps + amp_idx)
                save(n_count, n_st)

                # Reset all qubits
                if config.reset_type == "thermal":
                    for qubit in qubits:
                        qubit.wait(machine.thermalization_time * u.ns)
                elif config.reset_type == "active":
                    from quam_libs.macros import active_reset
                    for qubit in qubits:
                        active_reset(qubit, config.readout_operation)

                # Align all qubits
                align()

                # Play pulse on selected qubit (one-hot)
                for i, qubit in enumerate(qubits):
                    with if_(qubit_idx == i):
                        qubit.xy.play(config.drive_operation, amplitude_scale=amp_scale)

                # Align before measurement
                align()

                # Measure all qubits
                for i, qubit in enumerate(qubits):
                    qubit.resonator.measure(
                        config.readout_operation,
                        qua_vars=(I[i], Q[i])
                    )

                # Save IQ data
                for i in range(num_qubits):
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])

                # Wait for resonator depletion
                align()
                for qubit in qubits:
                    wait(qubit.resonator.depletion_time * u.ns, qubit.resonator.name)

    # Stream processing
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            # Buffer: shots → amplitudes → drive_qubits
            I_st[i].buffer(config.num_averages).buffer(config.n_amp_steps).buffer(num_qubits).save(f"I{i+1}")
            Q_st[i].buffer(config.num_averages).buffer(config.n_amp_steps).buffer(num_qubits).save(f"Q{i+1}")

print(f"  QUA program compiled successfully")
print(f"  Estimated runtime: ~{n_pulses * config.num_averages * 0.0001:.1f} minutes")

# %% Execute
if config.simulate:
    print("\n[5/6] Running simulation...")
    simulation_config = SimulationConfig(duration=config.simulation_duration_ns * 4)
    job = qmm.simulate(qua_config, amplitude_sweep, simulation_config)
    print("  Simulation complete")
else:
    print("\n[5/6] Executing on hardware...")
    print("  This will take a while...")

    with qm_session(qmm, qua_config, timeout=config.timeout) as qm:
        job = qm.execute(amplitude_sweep, timeout=config.timeout)

        # Progress tracking
        results = fetching_tool(job, ["n"], mode="live")
        total_measurements = n_pulses * config.num_averages

        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, total_measurements, start_time=results.start_time)

# %% Fetch and structure data
print("\n[6/6] Fetching and structuring data...")

if not config.simulate:
    # Fetch results
    result_handles = job.result_handles

    # Build xarray dataset
    data_vars = {}
    coords = {
        "drive_qubit": qubit_names,
        "measured_qubit": qubit_names,
        "amplitude_factor": amp_factors,
        "shot": np.arange(config.num_averages),
    }

    # Fetch IQ data for each measured qubit
    for i, qubit in enumerate(qubits):
        I_data = result_handles.get(f"I{i+1}").fetch_all()
        Q_data = result_handles.get(f"Q{i+1}").fetch_all()

        # Shape: (num_qubits, n_amp_steps, num_averages)
        data_vars[f"I_{qubit.name}"] = (["drive_qubit", "amplitude_factor", "shot"], I_data)
        data_vars[f"Q_{qubit.name}"] = (["drive_qubit", "amplitude_factor", "shot"], Q_data)

    # Create dataset
    ds = xr.Dataset(data_vars, coords=coords)

    # Add metadata
    ds.attrs["timestamp"] = datetime.now().isoformat()
    ds.attrs["num_qubits"] = num_qubits
    ds.attrs["n_amp_steps"] = config.n_amp_steps
    ds.attrs["num_averages"] = config.num_averages
    ds.attrs["drive_operation"] = config.drive_operation
    ds.attrs["amp_range"] = f"{config.min_amp_factor} to {config.max_amp_factor}"

    # Convert to volts
    conversion_factor = 2**12
    for qubit in qubits:
        readout_len = qubit.resonator.operations[config.readout_operation].length
        ds[f"I_{qubit.name}_V"] = ds[f"I_{qubit.name}"] / conversion_factor / readout_len
        ds[f"Q_{qubit.name}_V"] = ds[f"Q_{qubit.name}"] / conversion_factor / readout_len

    print(f"  Data shape: {num_qubits} drive qubits × {config.n_amp_steps} amps × {config.num_averages} shots × {num_qubits} measured qubits")

    # %% Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as NetCDF
    save_path_nc = config.save_dir / f"{timestamp}_amplitude_sweep_dataset.nc"
    config.save_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(save_path_nc)
    print(f"\nSaved NetCDF: {save_path_nc}")

    # Save as HDF5 (required format spec)
    save_path_h5 = config.save_dir / f"{timestamp}_amplitude_sweep_dataset.h5"
    with h5py.File(save_path_h5, 'w') as f:
        # ===== METADATA GROUP =====
        metadata = f.create_group('metadata')
        metadata.attrs['qubits'] = json.dumps(qubit_names)
        metadata.attrs['sample_rate_ns'] = 4.0
        metadata.attrs['n_shots'] = config.num_averages
        metadata.attrs['timestamp'] = timestamp
        metadata.attrs['notes'] = (
            f"Amplitude sweep dataset (one-hot encoding). "
            f"Each qubit driven individually across {config.n_amp_steps} amplitude steps "
            f"from {config.min_amp_factor}× to {config.max_amp_factor}× nominal. "
            f"{config.num_averages} shots per amplitude point. "
            f"N_time=1 (integrated IQ values)."
        )

        # ===== PULSE PARAMETER TABLE =====
        dt = np.dtype([
            ('drive_qubit_idx', 'i4'),
            ('pulse_type', 'S20'),
            ('amplitude', 'f8'),
            ('amplitude_factor', 'f8'),
            ('duration_ns', 'f8'),
            ('detuning_Hz', 'f8'),
            ('phase_rad', 'f8'),
            ('drag_alpha', 'f8'),
            ('sigma_ns', 'f8'),
            ('reserved1', 'f8'),
            ('reserved2', 'f8'),
        ])

        pulse_table = np.zeros(n_pulses, dtype=dt)
        for i, cfg in enumerate(pulse_configs):
            pulse_table[i] = (
                cfg['drive_qubit_idx'],
                config.drive_operation.encode('utf-8'),
                cfg['amplitude'],
                cfg['amplitude_factor'],
                cfg['duration_ns'],
                0.0,  # detuning_Hz
                0.0,  # phase_rad
                cfg['drag_alpha'],
                cfg['sigma_ns'],
                0.0,  # reserved1
                0.0,  # reserved2
            )

        f.create_dataset('drive_configs', data=pulse_table)

        # ===== RAW IQ MEASUREMENTS =====
        # Shape: [N_pulses, N_qubits, N_shots, N_time]
        I_array = np.zeros((n_pulses, num_qubits, config.num_averages, 1))
        Q_array = np.zeros((n_pulses, num_qubits, config.num_averages, 1))

        for measured_qubit_idx, qubit in enumerate(qubits):
            # Extract data: shape (num_drive_qubits, n_amp_steps, num_averages)
            I_data = ds[f"I_{qubit.name}"].values
            Q_data = ds[f"Q_{qubit.name}"].values

            # Reshape to pulse-indexed format
            pulse_idx = 0
            for drive_q_idx in range(num_qubits):
                for amp_idx in range(config.n_amp_steps):
                    I_array[pulse_idx, measured_qubit_idx, :, 0] = I_data[drive_q_idx, amp_idx, :]
                    Q_array[pulse_idx, measured_qubit_idx, :, 0] = Q_data[drive_q_idx, amp_idx, :]
                    pulse_idx += 1

        raw_iq_group = f.create_group('raw_iq')
        raw_iq_group.create_dataset(
            'I',
            data=I_array,
            compression='gzip',
            compression_opts=4
        )
        raw_iq_group.create_dataset(
            'Q',
            data=Q_array,
            compression='gzip',
            compression_opts=4
        )

        raw_iq_group.attrs['axis_0'] = 'pulse_index'
        raw_iq_group.attrs['axis_1'] = 'qubit_index'
        raw_iq_group.attrs['axis_2'] = 'shot_index'
        raw_iq_group.attrs['axis_3'] = 'time_sample (integrated: always 1)'
        raw_iq_group.attrs['shape'] = str(I_array.shape)

    print(f"Saved HDF5: {save_path_h5}")

    # %% Visualization
    print("\n" + "=" * 80)
    print("DATA COLLECTION COMPLETE")
    print("=" * 80)

    # Create visualization
    fig, axes = plt.subplots(num_qubits, num_qubits, figsize=(16, 16))

    for drive_idx, drive_qubit in enumerate(qubits):
        for meas_idx, meas_qubit in enumerate(qubits):
            ax = axes[meas_idx, drive_idx]

            # Get mean IQ across shots
            I_mean = ds[f"I_{meas_qubit.name}"].sel(drive_qubit=drive_qubit.name).mean(dim="shot")
            Q_mean = ds[f"Q_{meas_qubit.name}"].sel(drive_qubit=drive_qubit.name).mean(dim="shot")

            # Get absolute amplitudes
            base_amp = drive_qubit.xy.operations[config.drive_operation].amplitude
            abs_amps = amp_factors * base_amp * 1000  # mV

            # Plot IQ vs amplitude
            ax.plot(abs_amps, I_mean * 1000, 'b-', label='I', alpha=0.7)
            ax.plot(abs_amps, Q_mean * 1000, 'r-', label='Q', alpha=0.7)

            ax.set_xlabel(f"Drive amplitude (mV)")
            ax.set_ylabel("IQ (mV)")
            ax.set_title(f"Drive: {drive_qubit.name} → Meas: {meas_qubit.name}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    plt.suptitle(f"Amplitude Sweep - {timestamp}", fontsize=14)
    plt.tight_layout()

    fig_path = config.save_dir / f"{timestamp}_amplitude_sweep.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure: {fig_path}")

    plt.show()

    print("\nDataset Summary:")
    print(f"  Qubits: {qubit_names}")
    print(f"  Amplitude steps: {config.n_amp_steps} ({config.min_amp_factor}× to {config.max_amp_factor}×)")
    print(f"  Shots per point: {config.num_averages}")
    print(f"  Total configurations: {n_pulses}")
    print(f"  Total measurements: {n_pulses * config.num_averages}")
    print(f"\nData files:")
    print(f"  {save_path_nc}")
    print(f"  {save_path_h5}")
    print(f"  {fig_path}")

    print("\nDone!")
else:
    print("\nSimulation completed. No data saved.")
