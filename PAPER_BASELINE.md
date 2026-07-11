# Paper Baseline

This branch contains the paper-facing pair-energy baseline and manuscript.
The article is centered on fixed-temperature crystalline dynamics in a lattice
frame, rather than on temporal memory or general rotational transferability.

## Scope

Main article scope:

- conservative `pair-energy` ASE pipeline;
- `ref-plus-delta` pair input mode;
- one-frame MLP temporal encoder as the minimal headline model;
- fixed FCC orientation, volume, and temperature (`300 K`);
- a `3x3x3` local block with the central crystallographic cell as the joint
  force-prediction target;
- optimized full-crystal pair-scatter inference;
- standard 1055 ASE/NVT post-processing.

Excluded from this branch:

- `gram-pair` invariant input experiments;
- follow-up equivariance-specific search and inference runs;
- temporary experiment notes and local run artifacts.

Secondary controls:

- the frame-layered recurrent model is retained as a temporal-history ablation;
- the virial pressure anchor and elastic constants are supporting mechanical
  checks, not the primary article claim.

## Legacy Reference Model

The historical pre-equivariance recurrent reference model is:

```text
models333_pair_energy_frame_layered_refplusdelta_rl3_d30k_10_forceonly_20260622/
mean_norm_0.612829582889395_rnn_pair_energy_rnn_acceleration_h128_rl3_readoutfinalhidden_temporalframelayered_inputrefplusdelta_bidir_shells2_n18_targetforce_accnormglobal.pth
```

This model was used in the earlier long 1055 baseline runs, including:

```text
pair_energy_frame_layered_refplusdelta_sqw0613_1055_500000_bussi200_qinitial_only_nointernal_fastscatter
```

The corresponding long-run S(q,w) correlation was about:

```text
0.820729
```

## Reproducibility Status

The manuscript now uses the one-frame MLP `ref-plus-delta` model as its headline
configuration. The matching MLP implementation and integration-history slicing
are included in this branch. The historical checkpoint used for the 100 ps
headline run was:

```text
models333_pair_energy_mlp_refplusdelta_L1_d30k_10_forceonly_20260705/
mean_norm_0.6530284269166575_rnn_pair_energy_rnn_acceleration_h128_rl1_readoutfinalhidden_temporalmlp_inputrefplusdelta_bidir_shells2_n18_targetforce_accnormglobal.pth
```

The checkpoint itself is not stored in Git and is not currently present in the
local model archive. It must be recovered or the documented search must be
repeated before publication. The matching architecture has `67,841` trainable
parameters; record the final checkpoint checksum after recovery or retraining.

## Branch Intent

Use this branch for paper-facing material tied to the fixed-temperature,
pre-equivariance `ref-plus-delta` pair-energy model. Experimental invariant,
absolute-pair, signed-edge, and bond-vector-RBF input work should remain on
separate branches.
