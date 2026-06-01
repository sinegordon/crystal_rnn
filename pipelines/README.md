# Crystal RNN Pipelines

This folder contains the stable entrypoints for data preparation, model search,
ASE inference, cluster execution, and postprocessing.  The original root scripts
remain for compatibility, while `pipelines/` provides safer paradigm-specific
defaults.

## Paradigms

- `field/`: grid/field RNN models (`CrystalFieldRNNNet`).
- `pair_force/`: pair-local force models (`CrystalPairForceRNNNet`).
- `pair_energy/`: conservative pair-energy models (`CrystalPairEnergyRNNNet`).
- `shared/`: data preparation helpers, ASE wrappers, cluster helpers, and
  postprocessing shared by several paradigms.

## Current Recommended Path

The current working production path is `pair_energy`:

- train on force-enabled `3x3x3` Cu data;
- select by `S(q,w)` rollout score on `333`;
- run ASE 1055 NVT with Bussi thermostat;
- use direct model `E_tot = E_pot_model + E_kin` as the primary energy
  diagnostic;
- keep path-energy as a force/work consistency diagnostic, not as the main
  thermodynamic observable.

The clean ASE inference preset is:

- `T = 300 K`;
- `dt = 0.002 ps`;
- Bussi thermostat coupling `taut = 200 fs`;
- `q_zero_mode = none`;
- no history damping, adaptive eta, curl correction, or power-bias correction.

## End-to-End Pair-Energy Workflow

Prepare force-enabled crystal data from a LAMMPS dump:

```bash
python pipelines/pair_energy/data/prepare_force_data.py \
  /path/to/Cu333.LAMMPSDUMP \
  data333_force.npz \
  --input-format dump \
  --crystal-shape 3 3 3 \
  --train-supercell-shape 3 3 3 \
  --sequence-length 3 \
  --unit-cell-atoms 4 \
  --dt-ps 0.002 \
  --max-frames 100000
```

Submit a cluster model search:

```bash
python pipelines/pair_energy/cluster/submit_search.py \
  --run-label pair_energy_rnn_finalhidden_rl1_force_pmean_w01_aover_w01_aunder_w001_d90k_30 \
  --model-count 30 \
  --data-path data333_force.npz \
  --delta-frames 90000 \
  --epochs 100 \
  --batch-size 64 \
  --hidden-size 128 \
  --rnn-layers 1 \
  --rnn-readout-mode final-hidden \
  --training-target force \
  --acceleration-normalization global \
  --power-mean-loss-weight 0.1 \
  --acceleration-over-rms-loss-weight 0.1 \
  --acceleration-under-rms-loss-weight 0.01
```

Check the search:

```bash
python pipelines/pair_energy/cluster/check_search.py \
  --state-path logs/pair_energy_rnn_finalhidden_rl1_force_pmean_w01_aover_w01_aunder_w001_d90k_30_search_state.json
```

Run ASE inference on `1055` with the selected energy model:

```bash
python pipelines/pair_energy/cluster/submit_inference_1055.py \
  --model-path models333_pair_energy_rnn_finalhidden_rl1_force_pmean_w01_aover_w01_aunder_w001_d90k_30/mean_norm_0.6663052760722262_rnn_pair_energy_rnn_acceleration_h128_rl1_readoutfinalhidden_bidir_shells2_n18_targetforce_accnormglobal_pmean0.1_aover0.1_aunder0.01_op2_up2.pth \
  --data-path data1055.npz \
  --label pair_energy_best_1055_50000_bussi200_nointernal \
  --state-path logs/ase1055_pair_energy_best_50000_bussi200_nointernal_state.json \
  --steps 50000 \
  --taut-fs 200 \
  --q-zero-mode none \
  --history-damping-mode none \
  --temperature-eta-adaptive-mode none \
  --power-bias-correction-mode none
```

Fetch/check inference outputs:

```bash
python pipelines/pair_energy/cluster/fetch_inference_1055.py \
  --state-path logs/ase1055_pair_energy_best_50000_bussi200_nointernal_state.json
```

The fetch command downloads only lightweight postprocessing artifacts: PNG,
TXT, TSV, and logs.  It intentionally does not download large `.npz`
trajectories.

## Standard Inference Outputs

Cluster ASE inference writes:

- `ase_nvt_1055_sqw.png` and `.txt`: `S(q,w)` comparison.
- `ase_nvt_1055_temperature_trace.png` and `.txt`: temperature stability.
- `ase_nvt_1055_velocity_histograms.png` and `.txt`: velocity distribution.
- `ase_nvt_1055_phase_power_acceleration.png` and `.tsv`: phase-window power
  and acceleration diagnostics.
- `ase_nvt_1055_canonical_checks.png` and `.tsv`: canonical distribution
  checks.
- `ase_nvt_1055_sound_speed_ox.png` and `.tsv`: longitudinal sound speed along
  OX.
- `ase_nvt_1055_path_energy.png` and `.tsv`: path-energy force/work diagnostic.
- `ase_nvt_1055_heat_capacity.png` and `.tsv`: path-energy heat capacity
  diagnostic.
- `ase_nvt_1055_etot_trace.png` and `.tsv`: direct model `E_tot` drift.
- `ase_nvt_1055_etot_heat_capacity.png` and `.tsv`: direct model `E_tot`
  heat-capacity diagnostic.
- `ase_nvt_1055_etot_cumulative.png` and `.tsv`: cumulative direct `E_tot`
  running-mean and centered-integral diagnostics.

## Compatibility

Root scripts are still present because existing SLURM jobs and old commands
refer to them directly.  Prefer `pipelines/...` entrypoints for new work.  Once
the layout settles, root scripts can be reduced to compatibility wrappers in a
separate cleanup.
