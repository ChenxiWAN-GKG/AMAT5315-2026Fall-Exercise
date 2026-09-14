use md::{
    EnergySample, ForwardEuler, VelocityVerlet, initial_state, lennard_jones_energy, run_experiment,
};
use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const IMAGE_WIDTH: u32 = 1440;
const IMAGE_HEIGHT: u32 = 1440;
const PANEL_TOP: f64 = 180.0;
const PANEL_WIDTH: f64 = 570.0;
const PANEL_HEIGHT: f64 = 900.0;
const LEFT_PANEL_X: f64 = 90.0;
const RIGHT_PANEL_X: f64 = 780.0;
const ORANGE: &str = "#d97706";
const BLUE: &str = "#2563eb";
const GRID: &str = "#d1d5db";
const AXIS: &str = "#374151";

#[derive(Debug, Clone, Copy)]
struct PlotPoint {
    time: f64,
    value: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let output_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("week2/dimer.png"));

    let euler_samples = run_experiment(ForwardEuler, initial_state(), 0.01, 500)?;
    let verlet_samples = run_experiment(VelocityVerlet, initial_state(), 0.01, 5000)?;
    let reference_energy = lennard_jones_energy(1.2);

    let euler_points = relative_points(&euler_samples, reference_energy);
    let verlet_points = relative_points(&verlet_samples, reference_energy);
    let svg = build_svg(&euler_points, &verlet_points);
    write_png(&output_path, &svg)?;

    println!("Wrote {}", output_path.display());
    Ok(())
}

fn relative_points(samples: &[EnergySample], reference_energy: f64) -> Vec<PlotPoint> {
    samples
        .iter()
        .map(|sample| PlotPoint {
            time: sample.time,
            value: relative_error(sample.total_energy, reference_energy),
        })
        .collect()
}

fn relative_error(energy: f64, reference_energy: f64) -> f64 {
    (energy - reference_energy) / reference_energy.abs()
}

fn build_svg(euler_points: &[PlotPoint], verlet_points: &[PlotPoint]) -> String {
    let left_verlet: Vec<_> = verlet_points
        .iter()
        .copied()
        .filter(|point| point.time <= 5.0)
        .collect();
    let right_verlet: Vec<_> = verlet_points
        .iter()
        .copied()
        .map(|point| PlotPoint {
            time: point.time,
            value: point.value * 1000.0,
        })
        .collect();

    let left_range = y_range(
        euler_points
            .iter()
            .chain(left_verlet.iter())
            .map(|point| point.value),
    );
    let right_range = y_range(
        right_verlet
            .iter()
            .map(|point| point.value)
            .chain([-1.0, 1.0]),
    );

    let mut svg = String::new();
    writeln!(
        svg,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{IMAGE_WIDTH}" height="{IMAGE_HEIGHT}" viewBox="0 0 {IMAGE_WIDTH} {IMAGE_HEIGHT}">"##
    )
    .unwrap();
    svg.push_str(
        r##"<title>Two-atom molecular dynamics relative energy error</title>
<desc>Forward Euler and velocity-Verlet relative energy errors are compared over 500 steps. A longer velocity-Verlet run is shown at one-thousand times its signed relative error.</desc>
<rect width="100%" height="100%" fill="#ffffff"/>"##,
    );
    writeln!(
        svg,
        r##"<text x="{:.1}" y="42" font-family="Helvetica, Arial, sans-serif" font-size="26" font-weight="600" fill="#111827">Two-atom energy conservation</text>"##,
        LEFT_PANEL_X
    )
    .unwrap();
    writeln!(
        svg,
        r##"<text x="{:.1}" y="70" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#4b5563">Reduced Lennard–Jones units · dt = 0.01 · signed relative error</text>"##,
        LEFT_PANEL_X
    )
    .unwrap();

    append_panel(
        &mut svg,
        LEFT_PANEL_X,
        "500 steps: both integrators",
        "relative energy error, δE",
        5.0,
        left_range,
        &[(&euler_points, ORANGE), (&left_verlet, BLUE)],
        false,
    );
    append_panel(
        &mut svg,
        RIGHT_PANEL_X,
        "5000 steps: velocity-Verlet",
        "1000 × δE",
        50.0,
        right_range,
        &[(&right_verlet, BLUE)],
        true,
    );

    let legend_y = 1220.0;
    writeln!(
        svg,
        r##"<line x1="{:.1}" y1="{legend_y}" x2="{:.1}" y2="{legend_y}" stroke="{ORANGE}" stroke-width="3"/><text x="{:.1}" y="{}" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#374151">Forward Euler</text>"##,
        500.0,
        530.0,
        540.0,
        legend_y + 5.0
    )
    .unwrap();
    writeln!(
        svg,
        r##"<line x1="{:.1}" y1="{legend_y}" x2="{:.1}" y2="{legend_y}" stroke="{BLUE}" stroke-width="3"/><text x="{:.1}" y="{}" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#374151">Velocity-Verlet</text>"##,
        690.0,
        720.0,
        730.0,
        legend_y + 5.0
    )
    .unwrap();
    svg.push_str("</svg>\n");
    svg
}

fn append_panel(
    svg: &mut String,
    left: f64,
    title: &str,
    y_label: &str,
    maximum_time: f64,
    y_limits: (f64, f64),
    series: &[(&[PlotPoint], &str)],
    show_thresholds: bool,
) {
    let bottom = PANEL_TOP + PANEL_HEIGHT;
    writeln!(
        svg,
        r##"<text x="{:.1}" y="{}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="16" font-weight="600" fill="#111827">{title}</text>"##,
        left + PANEL_WIDTH / 2.0,
        PANEL_TOP - 22.0
    )
    .unwrap();
    writeln!(
        svg,
        r##"<rect x="{left}" y="{PANEL_TOP}" width="{PANEL_WIDTH}" height="{PANEL_HEIGHT}" fill="#ffffff" stroke="{AXIS}" stroke-width="1.5"/>"##
    )
    .unwrap();

    let y_zero = data_to_pixel(0.0, 0.0, maximum_time, y_limits, left).1;
    writeln!(
        svg,
        r##"<line x1="{left}" y1="{y_zero:.2}" x2="{}" y2="{y_zero:.2}" stroke="#9ca3af" stroke-dasharray="6 5"/>"##,
        left + PANEL_WIDTH
    )
    .unwrap();

    if show_thresholds {
        for threshold in [-1.0, 1.0] {
            let y = data_to_pixel(0.0, threshold, maximum_time, y_limits, left).1;
            writeln!(
                svg,
                r##"<line x1="{left}" y1="{y:.2}" x2="{}" y2="{y:.2}" stroke="#9ca3af" stroke-dasharray="3 4"/><text x="{}" y="{:.2}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#6b7280">{threshold:.0}</text>"##,
                left + PANEL_WIDTH,
                left + PANEL_WIDTH - 6.0,
                y - 4.0
            )
            .unwrap();
        }
    }

    for index in 0..=5 {
        let time = maximum_time * f64::from(index) / 5.0;
        let (x, _) = data_to_pixel(time, 0.0, maximum_time, y_limits, left);
        writeln!(
            svg,
            r##"<line x1="{x:.2}" y1="{PANEL_TOP}" x2="{x:.2}" y2="{bottom}" stroke="{GRID}" stroke-width="0.8"/><text x="{x:.2}" y="{}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#374151">{time:.1}</text>"##,
            bottom + 22.0
        )
        .unwrap();
    }

    for index in 0..=4 {
        let value = y_limits.0 + (y_limits.1 - y_limits.0) * f64::from(index) / 4.0;
        let (_, y) = data_to_pixel(0.0, value, maximum_time, y_limits, left);
        writeln!(
            svg,
            r##"<line x1="{left}" y1="{y:.2}" x2="{}" y2="{y:.2}" stroke="{GRID}" stroke-width="0.8"/><text x="{}" y="{:.2}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#374151">{value:.3e}</text>"##,
            left + PANEL_WIDTH,
            left - 8.0,
            y + 4.0
        )
        .unwrap();
    }

    for (points, color) in series {
        let path = path_for(points, maximum_time, y_limits, left);
        if !path.is_empty() {
            writeln!(
                svg,
                r##"<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>"##
            )
            .unwrap();
        }
    }

    writeln!(
        svg,
        r##"<text x="{}" y="{}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#111827">time</text><text x="{}" y="{}" text-anchor="middle" transform="rotate(-90 {} {})" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#111827">{y_label}</text>"##,
        left + PANEL_WIDTH / 2.0,
        bottom + 53.0,
        left - 56.0,
        PANEL_TOP + PANEL_HEIGHT / 2.0,
        left - 56.0,
        PANEL_TOP + PANEL_HEIGHT / 2.0
    )
    .unwrap();
}

fn path_for(points: &[PlotPoint], maximum_time: f64, y_limits: (f64, f64), left: f64) -> String {
    let mut path = String::new();
    for (index, point) in points.iter().enumerate() {
        if point.time > maximum_time {
            continue;
        }
        let (x, y) = data_to_pixel(point.time, point.value, maximum_time, y_limits, left);
        if index == 0 {
            write!(path, "M {x:.2} {y:.2}").unwrap();
        } else {
            write!(path, " L {x:.2} {y:.2}").unwrap();
        }
    }
    path
}

fn data_to_pixel(
    time: f64,
    value: f64,
    maximum_time: f64,
    y_limits: (f64, f64),
    left: f64,
) -> (f64, f64) {
    let x = left + time / maximum_time * PANEL_WIDTH;
    let fraction = (value - y_limits.0) / (y_limits.1 - y_limits.0);
    let y = PANEL_TOP + (1.0 - fraction) * PANEL_HEIGHT;
    (x, y)
}

fn y_range(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let values: Vec<_> = values.filter(|value| value.is_finite()).collect();
    if values.is_empty() {
        return (-1.0, 1.0);
    }

    let minimum = values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .min(0.0);
    let maximum = values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);
    let padding = ((maximum - minimum) * 0.08).max(1e-6);
    (minimum - padding, maximum + padding)
}

fn write_png(output_path: &Path, svg: &str) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let temporary_svg = env::temp_dir().join(format!("amat5315-dimer-{}.svg", std::process::id()));
    fs::write(&temporary_svg, svg)?;

    let conversion = Command::new("/usr/bin/qlmanage")
        .args(["-t", "-s", &IMAGE_WIDTH.to_string(), "-o"])
        .arg(env::temp_dir())
        .arg(&temporary_svg)
        .output()?;
    if !conversion.status.success() {
        let message = String::from_utf8_lossy(&conversion.stderr);
        let _ = fs::remove_file(&temporary_svg);
        return Err(std::io::Error::other(format!(
            "Quick Look could not create the PNG: {message}"
        ))
        .into());
    }

    let generated_png = temporary_svg.with_extension("svg.png");
    fs::copy(&generated_png, output_path)?;
    fs::remove_file(&temporary_svg)?;
    fs::remove_file(generated_png)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_error_uses_absolute_reference_energy() {
        let reference = -0.9;

        assert!((relative_error(-0.8, reference) - (0.1 / 0.9)).abs() < 1e-12);
    }

    #[test]
    fn svg_has_requested_panel_labels_and_colors() {
        let svg = build_svg(&[], &[]);

        assert!(svg.contains("relative energy error, δE"));
        assert!(svg.contains("1000 × δE"));
        assert!(svg.contains("#d97706"));
        assert!(svg.contains("#2563eb"));
    }
}
