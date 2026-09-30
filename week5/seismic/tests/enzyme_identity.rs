use seismic::solver::{enzyme_step_cell, enzyme_step_cell_reverse};

#[test]
fn local_enzyme_jvp_and_vjp_satisfy_dot_product_identity() {
    let primal = [0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 1.8];
    let tangent = [0.11, -0.13, 0.17, -0.19, 0.23, -0.29, 0.07];
    let constants = (0.4, 0.8, 1.0, 0.2);
    let output_bar = -0.37;
    let (_, output_tangent) = enzyme_step_cell(
        primal[0],
        primal[1],
        primal[2],
        primal[3],
        primal[4],
        primal[5],
        primal[6],
        constants.0,
        constants.1,
        constants.2,
        constants.3,
        tangent[0],
        tangent[1],
        tangent[2],
        tangent[3],
        tangent[4],
        tangent[5],
        tangent[6],
    );
    let bars = enzyme_step_cell_reverse(
        primal[0],
        primal[1],
        primal[2],
        primal[3],
        primal[4],
        primal[5],
        primal[6],
        constants.0,
        constants.1,
        constants.2,
        constants.3,
        output_bar,
    );
    let right = bars[1..8]
        .iter()
        .zip(tangent)
        .map(|(bar, direction)| bar * direction)
        .sum::<f64>();
    assert!((output_bar * output_tangent - right).abs() < 1e-12);
}
