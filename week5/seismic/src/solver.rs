use std::f64::consts::PI;

use ndarray::Array2;

pub fn build_sponge(nx: usize, nz: usize, width: f64, strength: f64) -> Array2<f64> {
    let mut sponge = Array2::zeros((nz, nx));
    if width <= 0.0 {
        return sponge;
    }
    for z in 0..nz {
        for x in 0..nx {
            let distance = (x.min(nx - 1 - x))
                .min(z.min(nz - 1 - z)) as f64;
            let factor = (1.0 - distance / width).max(0.0);
            sponge[[z, x]] = strength * factor * factor;
        }
    }
    sponge
}

pub fn ricker(t: f64, frequency: f64, peak_time: f64) -> f64 {
    let theta = PI * frequency * (t - peak_time);
    (1.0 - 2.0 * theta * theta) * (-theta * theta).exp()
}

pub fn gaussian_footprint(
    nx: usize,
    nz: usize,
    shot_x: usize,
    shot_z: usize,
) -> Array2<f64> {
    let mut footprint = Array2::zeros((nz, nx));
    for z in 0..nz {
        for x in 0..nx {
            let dx = x as f64 - shot_x as f64;
            let dz = z as f64 - shot_z as f64;
            footprint[[z, x]] = (-(dx * dx + dz * dz) / 2.0).exp();
        }
    }
    footprint
}

pub fn advance(
    previous: &Array2<f64>,
    current: &Array2<f64>,
    next: &mut Array2<f64>,
    speed: &Array2<f64>,
    sigma: &Array2<f64>,
    source: &Array2<f64>,
    dx: f64,
    dt: f64,
) {
    next.fill(0.0);
    let (nz, nx) = current.dim();
    for z in 1..(nz - 1) {
        for x in 1..(nx - 1) {
            let laplacian = (current[[z, x - 1]]
                + current[[z, x + 1]]
                + current[[z - 1, x]]
                + current[[z + 1, x]]
                - 4.0 * current[[z, x]])
                / (dx * dx);
            next[[z, x]] = (2.0 * current[[z, x]]
                - (1.0 - sigma[[z, x]] * dt) * previous[[z, x]]
                + dt * dt
                    * (speed[[z, x]] * speed[[z, x]] * laplacian + source[[z, x]]))
                / (1.0 + sigma[[z, x]] * dt);
        }
    }
}

pub fn sample_receivers(field: &Array2<f64>, receivers: &[[usize; 2]]) -> Vec<f64> {
    receivers
        .iter()
        .map(|&[x, z]| field[[z, x]])
        .collect()
}
