# Plan: energy architecture

This note records a postponed route for making the learned dynamics more
physically stable.

## Motivation

The current acceleration RNN can produce a small systematic positive mean power
`<a * v>` during long ASE rollouts.  The history-damping `eta` correction
removes one visible local anti-damping channel, but diagnostics show it does not
capture all of the long-time energy drift.

## Key Point

An energy-based architecture does not strictly require energy labels.  A model
can learn a potential-like scalar `U_theta(u)` and be trained only from forces
or accelerations:

```text
a_theta(u) = -grad U_theta(u)
L = MSE(a_theta(u), a_ref)
```

Energy labels would help fix the potential scale and shape, but the dynamics
primarily needs gradients, so force-only training is possible.

## Candidate Direction

1. Build a local energy model from pair or neighborhood geometric features.
2. Obtain accelerations by automatic differentiation of the scalar energy.
3. Train against force-derived discrete accelerations from prepared `.npz`
   files.
4. Keep optional RNN/history corrections only as a constrained residual:
   `a = -grad U_theta(u_t) + a_history`.
5. Penalize the residual power so that the non-potential part cannot become a
   long-time heat source.

## Why Not Now

This is a larger architectural change.  The current immediate experiment is a
cheaper inference-side power-bias correction on top of the existing edge RNN and
history-damping `eta` mechanism.
