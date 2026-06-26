# Pair-Energy Pipeline

Pair-energy models learn a local scalar pair energy and obtain forces by
differentiating it.  This is the conservative pair paradigm.

## Main Entrypoints

- `data/prepare_force_data.py`: prepare force-enabled crystal `.npz` data.
- `search/find_models.py`: pair-energy model search.
- `cluster/submit_search.py`: submit pair-energy search to the cluster.
- `cluster/check_search.py`: inspect pair-energy search progress.
- `cluster/submit_inference_1055.py`: submit ASE 1055 inference.
- `cluster/fetch_inference_1055.py`: check/fetch ASE 1055 inference outputs.
- `postprocess/plot_etot_heat_capacity.py`: direct `E_tot` heat-capacity
  diagnostic for energy-aware ASE outputs.

## Architecture References

- `docs/ENERGY_MODEL_ARCHITECTURE.docx`: architecture description document.
- `docs/PAIR_ENERGY_ARCHITECTURE.svg`: editable vector architecture diagram.
- `docs/PAIR_ENERGY_ARCHITECTURE.png`: rendered architecture diagram.

## Recommended Inference Defaults

The current best-tested path is a clean Bussi NVT run:

```bash
python pipelines/pair_energy/cluster/submit_inference_1055.py \
  --model-path models333_pair_energy_rnn_finalhidden_rl1_force_pmean_w01_aover_w01_aunder_w001_d90k_30/mean_norm_0.6663052760722262_rnn_pair_energy_rnn_acceleration_h128_rl1_readoutfinalhidden_bidir_shells2_n18_targetforce_accnormglobal_pmean0.1_aover0.1_aunder0.01_op2_up2.pth \
  --data-path data1055.npz \
  --label pair_energy_best_1055_50000_bussi200_nointernal \
  --state-path logs/ase1055_pair_energy_best_50000_bussi200_nointernal_state.json \
  --steps 50000 \
  --dt-ps 0.002 \
  --temperature-k 300 \
  --taut-fs 200 \
  --q-zero-mode initial \
  --history-damping-mode none \
  --temperature-eta-adaptive-mode none \
  --power-bias-correction-mode none
```

Then fetch the lightweight results:

```bash
python pipelines/pair_energy/cluster/fetch_inference_1055.py \
  --state-path logs/ase1055_pair_energy_best_50000_bussi200_nointernal_state.json
```

The most important energy-aware outputs are:

- `ase_nvt_1055_etot_trace.png`: direct `E_tot` fluctuations and drift.
- `ase_nvt_1055_etot_heat_capacity.png`: direct `E_tot` heat capacity.
- `ase_nvt_1055_etot_cumulative.png`: cumulative direct `E_tot` checks.

Path-energy plots are still generated, but for pair-energy models they are
secondary diagnostics of force/work consistency rather than the main
thermodynamic observable.

`--q-zero-mode initial` removes the initial spatial q=0 velocity from both the
RNN history and ASE velocities.  It does not project the trajectory after every
MD step.  Use `initial-every-step`, `zero-every-step`, or
`constant-velocity-every-step` only for explicit constrained-control tests.

## Experimental Frame-Layered Temporal Core

The pair-energy search can use a frame-layered recurrent core:

```bash
python pipelines/pair_energy/cluster/submit_search.py \
  --temporal-architecture frame-layered \
  --rnn-layers 3
```

In this mode each history frame is assigned to its own recurrent cell instead
of passing every frame through every stacked PyTorch RNN layer.  Therefore
`--rnn-layers` must be exactly equal to the recurrent input sequence length.

## Experimental Relative-To-First Temporal Input

The experimental relative-history mode still reads three raw history frames,
but the RNN receives only two recurrent steps:

- step 1: raw frame 1 minus raw frame 0;
- step 2: raw frame 2 minus raw frame 0.

Pair features are then built from these relative changes without adding the
equilibrium pair vector.  The equilibrium geometry is used only to choose the
fixed nearest-neighbor coordination stencil.  This keeps the existing
`--neighbor-shells`/`--cutoff-scale` neighbor selection and allows raw triplets
to store either displacements or coordinates.

For a frame-layered pair-energy search on three-frame datasets, use two
frame layers because only two processed frames enter the recurrent block:

```bash
python pipelines/pair_energy/cluster/submit_search.py \
  --temporal-architecture frame-layered \
  --temporal-input-mode relative-to-first \
  --rnn-layers 2
```

## Experimental Ref-Plus-Delta Pair Input

The `ref-plus-delta` input keeps the usual history length but separates the
large equilibrium pair vector from the small dynamic displacement difference.
Each pair/time sample has six channels:

- `R_ref/a0`;
- `(u_neighbor-u_center)/a0`.

For pair-energy models, forces are differentiated only through the dynamic
displacement channels.  A search can be submitted with:

```bash
python pipelines/pair_energy/cluster/submit_search.py \
  --temporal-input-mode ref-plus-delta \
  --run-label pair_energy_ref_plus_delta_test
```
