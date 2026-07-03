# Paper Baseline

This branch records the pair-energy pipeline state used as the paper baseline
before the rotational-invariance / equivariance experiments.

## Scope

Included:

- conservative `pair-energy` ASE pipeline;
- frame-layered recurrent temporal core;
- `ref-plus-delta` pair input mode;
- optimized full-crystal pair-scatter inference;
- standard 1055 ASE/NVT post-processing.

Excluded from this branch:

- `gram-pair` invariant input experiments;
- follow-up equivariance-specific search and inference runs;
- temporary experiment notes and local run artifacts.

## Reference Model

The main pre-equivariance reference model is:

```text
models333_pair_energy_frame_layered_refplusdelta_rl3_d30k_10_forceonly_20260622/
mean_norm_0.612829582889395_rnn_pair_energy_rnn_acceleration_h128_rl3_readoutfinalhidden_temporalframelayered_inputrefplusdelta_bidir_shells2_n18_targetforce_accnormglobal.pth
```

This model was used in the long 1055 baseline runs, including:

```text
pair_energy_frame_layered_refplusdelta_sqw0613_1055_500000_bussi200_qinitial_only_nointernal_fastscatter
```

The corresponding long-run S(q,w) correlation was about:

```text
0.820729
```

## Branch Intent

Use this branch for paper-facing material that should stay tied to the
pre-equivariance `ref-plus-delta` pair-energy model.  Experimental invariant
input work should remain on a separate branch.
