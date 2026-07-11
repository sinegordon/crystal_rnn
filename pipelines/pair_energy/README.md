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

## Paper Baseline

The paper baseline is a one-frame pair-energy MLP in the fixed FCC lattice
frame:

- temporal architecture: `mlp`;
- temporal input: `ref-plus-delta`;
- model frames: `1` (`--rnn-layers 1` is retained as a compatibility name);
- local context: `3x3x3` unit cells;
- joint prediction target: all four atoms of the central unit cell;
- neighbors: two FCC shells (`18` neighbors per central atom);
- training target: force-derived acceleration;
- temperature and reference state: `300 K` at the fixed reference cell;
- auxiliary loss weights: zero for the headline force-only model.

With `hidden_size=128` and the historical compatibility setting
`bidirectional=True` (which sets the MLP representation width to 256), the
model has `67,841` trainable parameters.

Prepared datasets can still store three history frames. The search code takes
only the latest frame for MLP1 training, while rollout and ASE retain enough
physical history to initialize velocity and Verlet integration.

### Search

The pair-energy wrappers now use the paper baseline by default:

```bash
python pipelines/pair_energy/search/find_models.py 10 RNN \
  --data-path data333_force.npz \
  --models-dir models333_pair_energy_mlp_refplusdelta_d30k \
  --delta-frames 30000 \
  --epochs 50 \
  --save-all
```

The `RNN` positional argument is retained for compatibility with the shared
search CLI and is ignored by the MLP temporal encoder.

The corresponding cluster submission is:

```bash
python pipelines/pair_energy/cluster/submit_search.py \
  --run-label paper_pair_energy_mlp_refplusdelta_d30k \
  --model-count 30
```

### Inference

Use the selected checkpoint in a clean Bussi NVT run:

```bash
python pipelines/pair_energy/cluster/submit_inference_1055.py \
  --model-path '<path-to-selected-refplusdelta-mlp-checkpoint.pth>' \
  --data-path data1055.npz \
  --label paper_refplusdelta_mlp_1055_50000_bussi200 \
  --state-path logs/paper_refplusdelta_mlp_1055_50000_state.json \
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
  --state-path logs/paper_refplusdelta_mlp_1055_50000_state.json
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

## Temporal-History Control

The article control experiment can use a frame-layered recurrent core:

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

## Ref-Plus-Delta Pair Input

The headline `ref-plus-delta` input separates the
large equilibrium pair vector from the small dynamic displacement difference.
Each pair/time sample has six channels:

- `R_ref/a0`;
- `(u_neighbor-u_center)/a0`.

For pair-energy models, forces are differentiated only through the dynamic
displacement channels. The MLP1 paper baseline is selected with:

```bash
python pipelines/pair_energy/cluster/submit_search.py \
  --temporal-architecture mlp \
  --temporal-input-mode ref-plus-delta \
  --rnn-layers 1 \
  --run-label pair_energy_ref_plus_delta_test
```

## Optional Reference-Pressure Anchor

The headline model is force-only and uses a zero pressure-loss weight. The
supporting mechanical experiment adds the reference-lattice condition described
in the manuscript:

```bash
python pipelines/pair_energy/cluster/submit_search.py \
  --run-label paper_refplusdelta_mlp_pref1e4 \
  --reference-pressure-loss-weight 10000 \
  --reference-pressure-target 0 \
  --reference-pressure-loss-scale 1
```

This term is evaluated at zero displacement in the fixed reference cell. It is
an optional mechanical anchor and is not enabled by the paper baseline defaults.
