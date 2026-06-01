# Shared Postprocessing

These entrypoints wrap the root plotting scripts used by ASE and direct rollout
diagnostics.

Common diagnostics:

- `plot_sqw_comparison.py`
- `plot_ase_velocity_histograms.py`
- `plot_ase_temperature_trace.py`
- `plot_ase_phase_histograms.py`
- `plot_ase_canonical_checks.py`
- `plot_ase_sound_speed_ox.py`
- `plot_ase_path_energy.py`
- `plot_ase_heat_capacity.py`

Energy-aware diagnostics:

- `plot_ase_total_energy.py`
- `plot_ase_etot_heat_capacity.py`
- `plot_ase_etot_cumulative.py`

`plot_ase_heat_capacity.py` always uses path-energy.  Direct `E_tot`
heat capacity and cumulative direct `E_tot` checks are intentionally separate
scripts.
