# MLP postprocessing

Run the complete plotting set with one command:

```bash
python pipelines/mlp/postprocess/run_all.py \
  --ase-path outputs/run/trajectory.npz \
  --data-path data1055.npz \
  --output-dir outputs/run/postprocess
```

The command creates the S(q,w), velocity, displacement, VACF, RDF,
temperature, COM, canonical, sound-speed, path-energy, direct-energy, and heat
capacity diagnostics used for article runs. Optional diagnostics that cannot be
computed from a short trajectory are reported without discarding the remaining
results. Use `--strict` to stop at the first failure.
