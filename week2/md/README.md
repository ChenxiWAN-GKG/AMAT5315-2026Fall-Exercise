# Molecular dynamics crate

This crate contains the two-atom Euler and velocity-Verlet example and the periodic Lennard-Jones fluid. The reproduction commands and evidence inventory are in [`../README.md`](../README.md); the argument and recording contract is [`../md.design.toml`](../md.design.toml).

`md` runs one simulation per invocation. It requires `--n`, `--rho`, `--temperature`, `--dt`, `--eq-steps`, `--steps`, `--sample-every`, `--seed`, and `--out`. `--force cells` is the default; `--force naive` retains the direct pair search for comparison. `--ramp-to` activates production heating. The diagnostics and video are separate scripts that read `run.json` and `traj.jsonl`.

The pair force is derived from the Lennard-Jones potential. Both force searches use the same pair interaction and minimum-image displacement. `velocity_verlet_step_with_method` chooses the search; it does not change the integration rule. Production without `--ramp-to` has no thermostat.
