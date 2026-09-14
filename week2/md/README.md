# Two-atom molecular dynamics example

From the repository root, reproduce the two-panel relative-energy-error plot with:

```sh
cargo run --manifest-path week2/md/Cargo.toml --example dimer -- week2/dimer.png
```

The example uses the shared forward Euler and velocity-Verlet simulation and
converts its generated SVG to PNG with macOS Quick Look.
