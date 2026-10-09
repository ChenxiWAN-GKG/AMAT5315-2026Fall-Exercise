# Week 3: two-dimensional Ising model

The `ising` command samples an L × L periodic lattice with spins +1 or −1, nearest-neighbour coupling J = 1, and no external field. A Metropolis step makes L² randomly chosen single-spin proposals; a Wolff step flips one connected cluster. The seed fixes the random stream. Both methods record every measured step, so the analysis uses the sampled states rather than a summary printed by the simulator.

The data contract is in `ising.design.toml`. From this directory, install the Rust command and use Python with NumPy and Matplotlib for the plots:

```sh
cargo install --path .
mkdir -p runs artifacts evidence
```

Generate the initial checks and viewer recording:

```sh
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/T1.8
ising --update metropolis --l 64 --t-from 3.0 --t-to 3.0 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/T3.0
ising --update metropolis --l 64 --t-from 3.1 --t-to 3.1 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/T3.1
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/a
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/b
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 --discard 2000 --measure 2000 --seed 2027 --out runs/c
diff runs/a/series.jsonl runs/b/series.jsonl
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.05 --discard 2000 --measure 200 --every 20 --seed 2026 --out runs/ramp
cp runs/ramp/spins.jsonl spins.jsonl
```

Generate the four Metropolis runs used for the temperature and uncertainty comparisons. Each window run records 100,000 sweeps at every temperature; keep `artifacts/` local.

```sh
ising --update metropolis --l 32 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 1042 --out artifacts/coarse-l32
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 42 --out artifacts/coarse-l64
ising --update metropolis --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 1042 --out artifacts/window-l32
ising --update metropolis --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 42 --out artifacts/window-l64
```

Generate the two Wolff runs with one recorded cluster flip per row:

```sh
ising --update wolff --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 42 --out artifacts/wolff-l64
ising --update wolff --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 1042 --out artifacts/wolff-l32
```

Run these commands from `week3/` after the corresponding data exist. The output column names every committed evidence file produced by each command.

| Command | Evidence output |
| --- | --- |
| `python3 evidence/plot_boltzmann.py` | `evidence/boltzmann.png` |
| `python3 evidence/plot_ising_observables.py` | `evidence/magnetization.png`, `evidence/susceptibility.png` |
| `python3 scripts/peaks.py` | `evidence/peaks.txt` |
| `python3 scripts/trace.py` | `evidence/trace.png` |
| `python3 scripts/errors.py` | `evidence/errors.txt` |
| `python3 scripts/acf_binning.py` | `evidence/acf-binning.png` |
| `python3 scripts/chi_bootstrap.py` | `evidence/chi-bootstrap.png`, `evidence/chi-bootstrap.txt` |
| `python3 -m scripts.compare` | `evidence/tau.png`, `evidence/magnetization-compare.png`, `evidence/tau-compare.png` |

For `evidence/viewer-T1.8.png`, `evidence/viewer-T2.3.png`, and `evidence/viewer-T3.0.png`, load the published raw `week3/spins.jsonl` in the [course viewer](https://giggleliu.github.io/AMAT5315-2026Fall/week3-viewer.html), select each named temperature, and use **save PNG**. The corresponding **Copy link** address should also load in a private browser window.

The susceptibility here is χ(T) = L²(⟨M²⟩ − ⟨|M|⟩²)/T. Verified from the recorded rows, a five-point quadratic fit around each peak gives the Metropolis estimates Tpeak(32) = 2.33553 and Tpeak(64) = 2.30633, so the two-size estimate 2 Tpeak(64) − Tpeak(32) is 2.27712. The Wolff estimates are 2.34920 and 2.31166, yielding 2.27413. Both lie within 2% of the course sheet's source-reported exact value 2.26919. At L = 64 and T = 2.3, the Wolff mean |M| is 0.43095 and the Metropolis mean is 0.42150. Their difference is 0.52 combined standard errors using 8,000-step blocks (sweeps for Metropolis, cluster moves for Wolff). The measured autocorrelation times are 4.71 cluster moves and 573.34 Metropolis sweeps. A Wolff move flips 921.64 spins on average, giving 1.06 L² spin updates per correlation time versus 573.34 for Metropolis, a factor of about 541.

## Limitations

The Metropolis magnetization error at T = 2.3 varies with bootstrap block length, so the agreement between samplers is provisional. The Wolff critical-temperature bootstrap errors at block lengths 2,000, 4,000, and 8,000 moves are 0.00068, 0.00073, and 0.00050; their variation leaves that sampling error unresolved. The two-size extrapolation also leaves finite-size and peak-fit bias. Spin-update work counts flipped or proposed spins; it is not elapsed runtime.
