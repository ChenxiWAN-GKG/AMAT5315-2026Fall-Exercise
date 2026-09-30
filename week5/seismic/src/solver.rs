use std::f64::consts::PI;

use ndarray::Array2;

unsafe extern "C" {
    #[link_name = "enzyme_step_cell_reverse"]
    fn enzyme_step_cell_reverse_ffi(
        previous: f64,
        current: f64,
        left: f64,
        right: f64,
        up: f64,
        down: f64,
        speed: f64,
        sigma: f64,
        source: f64,
        dx: f64,
        dt: f64,
        output_bar: f64,
        out: *mut f64,
    );

    fn enzyme_step_cell_forward(
        previous: f64,
        current: f64,
        left: f64,
        right: f64,
        up: f64,
        down: f64,
        speed: f64,
        sigma: f64,
        source: f64,
        dx: f64,
        dt: f64,
        previous_dot: f64,
        current_dot: f64,
        left_dot: f64,
        right_dot: f64,
        up_dot: f64,
        down_dot: f64,
        speed_dot: f64,
        out: *mut f64,
    );
}

pub fn enzyme_step_cell_reverse(
    previous: f64,
    current: f64,
    left: f64,
    right: f64,
    up: f64,
    down: f64,
    speed: f64,
    sigma: f64,
    source: f64,
    dx: f64,
    dt: f64,
    output_bar: f64,
) -> [f64; 8] {
    let mut output = [0.0; 8];
    unsafe {
        enzyme_step_cell_reverse_ffi(
            previous,
            current,
            left,
            right,
            up,
            down,
            speed,
            sigma,
            source,
            dx,
            dt,
            output_bar,
            output.as_mut_ptr(),
        );
    }
    output
}

pub fn enzyme_step_cell(
    previous: f64,
    current: f64,
    left: f64,
    right: f64,
    up: f64,
    down: f64,
    speed: f64,
    sigma: f64,
    source: f64,
    dx: f64,
    dt: f64,
    previous_dot: f64,
    current_dot: f64,
    left_dot: f64,
    right_dot: f64,
    up_dot: f64,
    down_dot: f64,
    speed_dot: f64,
) -> (f64, f64) {
    let mut output = [0.0; 2];
    unsafe {
        enzyme_step_cell_forward(
            previous,
            current,
            left,
            right,
            up,
            down,
            speed,
            sigma,
            source,
            dx,
            dt,
            previous_dot,
            current_dot,
            left_dot,
            right_dot,
            up_dot,
            down_dot,
            speed_dot,
            output.as_mut_ptr(),
        );
    }
    (output[0], output[1])
}

pub fn build_sponge(nx: usize, nz: usize, width: f64, strength: f64) -> Array2<f64> {
    let mut sponge = Array2::zeros((nz, nx));
    if width <= 0.0 {
        return sponge;
    }
    for z in 0..nz {
        for x in 0..nx {
            let distance = (x.min(nx - 1 - x)).min(z.min(nz - 1 - z)) as f64;
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

pub fn gaussian_footprint(nx: usize, nz: usize, shot_x: usize, shot_z: usize) -> Array2<f64> {
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
            next[[z, x]] = (2.0 * current[[z, x]] - (1.0 - sigma[[z, x]] * dt) * previous[[z, x]]
                + dt * dt * (speed[[z, x]] * speed[[z, x]] * laplacian + source[[z, x]]))
                / (1.0 + sigma[[z, x]] * dt);
        }
    }
}

pub fn sample_receivers(field: &Array2<f64>, receivers: &[[usize; 2]]) -> Vec<f64> {
    receivers.iter().map(|&[x, z]| field[[z, x]]).collect()
}
