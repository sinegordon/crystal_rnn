# RNN Conservative Correction Plan

This note records a postponed route for improving the equilibrium
configuration measure while keeping the recurrent model.

## Motivation

The pair-force RNN with fixed history damping can stabilize temperature and
velocity distributions, but canonical checks show that displacement moments and
q-resolved displacement/velocity mode powers can still differ from the
reference ensemble.

The goal is to preserve the useful RNN dynamics and add a more physically
constrained correction that can improve the configurational part of the
canonical ensemble.

## Proposed Route

Keep the current recurrent pair-force model as the base predictor:

```text
a_base = RNN(history)
```

Add a small conservative correction:

```text
a_eff = a_base - grad U_corr(u)
```

where `U_corr` is a learned scalar correction potential.  The correction should
be intentionally small and regularized, so it fixes equilibrium displacement
statistics without replacing the RNN dynamics.

## Candidate Losses

- Force/acceleration matching for the corrected acceleration.
- Displacement moment losses on rollout trajectories.
- q-mode losses for `<|u(q)|^2>` and optionally `<|v(q)|^2>`.
- Regularization on correction magnitude and spatial smoothness.

## Intended Order

Return to this after the adaptive `a*v` / `eta` correction is made stable enough
for long inference runs.
