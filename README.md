# Crystal RNN Pair-Energy Pipeline

This branch contains the focused pipeline for the conservative
`pair_energy` model only.  The public entrypoints live under
`pipelines/pair_energy/`; shared helpers are kept under `pipelines/shared/`.

The current production path is:

1. Prepare force-enabled Cu crystal data from a LAMMPS dump.
2. Train/search pair-energy RNN candidates on `3x3x3` blocks.
3. Select models by `S(q,w)` rollout and energy/velocity diagnostics.
4. Run ASE NVT inference on larger crystals, currently `1055`.
5. Fetch lightweight postprocessing outputs.

## Model

The pair-energy model is `CrystalPairEnergyRNNNet`.  It maps local histories of
relative pair vectors to an orientation-even scalar pair energy.  Forces are
obtained by differentiating that learned scalar with respect to the current
pair-vector features.  During full-crystal inference, each unique pair scatters
opposite contributions to the two atoms.

Architecture references:

- [ENERGY_MODEL_ARCHITECTURE.docx](pipelines/pair_energy/docs/ENERGY_MODEL_ARCHITECTURE.docx)
- [PAIR_ENERGY_ARCHITECTURE.svg](pipelines/pair_energy/docs/PAIR_ENERGY_ARCHITECTURE.svg)
- [PAIR_ENERGY_ARCHITECTURE.png](pipelines/pair_energy/docs/PAIR_ENERGY_ARCHITECTURE.png)

## Cluster Configuration

Cluster defaults are configured in:

```text
pipelines/shared/cluster/cluster_config.json
```

Every submit/check/fetch command also accepts `--cluster-config`, and the same
path can be supplied through `CRYSTAL_RNN_CLUSTER_CONFIG`.

## Prepare Data

Force-enabled data is required because pair-energy models are trained by force
targets, not by reference energies:

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

The wrapper forces `--include-forces` and `--require-forces`.

## Search Models

Recommended search command:

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

Check progress:

```bash
python pipelines/pair_energy/cluster/check_search.py \
  --state-path logs/pair_energy_rnn_finalhidden_rl1_force_pmean_w01_aover_w01_aunder_w001_d90k_30_search_state.json
```

## Run ASE Inference

The clean validation preset uses Bussi NVT, `T = 300 K`, `dt = 0.002 ps`,
`taut = 200 fs`, `q_zero_mode = none`, and no internal history/adaptive/power
correctors.

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

Fetch/check results:

```bash
python pipelines/pair_energy/cluster/fetch_inference_1055.py \
  --state-path logs/ase1055_pair_energy_best_50000_bussi200_nointernal_state.json
```

The fetch command downloads PNG, TXT, TSV, and logs only.  It intentionally
does not fetch large `.npz` trajectories.

## Standard Outputs

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
- `ase_nvt_1055_heat_capacity.png` and `.tsv`: path-energy heat-capacity
  diagnostic.
- `ase_nvt_1055_etot_trace.png` and `.tsv`: direct model `E_tot` drift.
- `ase_nvt_1055_etot_heat_capacity.png` and `.tsv`: direct model `E_tot`
  heat-capacity diagnostic.
- `ase_nvt_1055_etot_cumulative.png` and `.tsv`: cumulative direct `E_tot`
  running-mean and centered-integral diagnostics.

For pair-energy models, direct `E_tot = E_pot_model + E_kin` is the primary
energy diagnostic.  Path-energy is kept as a force/work consistency diagnostic.
