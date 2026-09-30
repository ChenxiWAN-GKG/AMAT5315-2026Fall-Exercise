#![no_std]
#![feature(autodiff)]

use core::autodiff::autodiff_forward;

#[autodiff_forward(
    step_cell_forward,
    Dual,
    Dual,
    Dual,
    Dual,
    Dual,
    Dual,
    Dual,
    Const,
    Const,
    Const,
    Const,
    Dual
)]
fn step_cell(
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
) -> f64 {
    let laplacian = (left + right + up + down - 4.0 * current) / (dx * dx);
    (2.0 * current - (1.0 - sigma * dt) * previous
        + dt * dt * (speed * speed * laplacian + source))
        / (1.0 + sigma * dt)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_step_cell_forward(
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
) {
    let (value, tangent) = step_cell_forward(
        previous,
        previous_dot,
        current,
        current_dot,
        left,
        left_dot,
        right,
        right_dot,
        up,
        up_dot,
        down,
        down_dot,
        speed,
        speed_dot,
        sigma,
        source,
        dx,
        dt,
    );
    unsafe {
        *out = value;
        *out.add(1) = tangent;
    }
}

unsafe extern "C" {
    fn abort() -> !;
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { abort() }
}
