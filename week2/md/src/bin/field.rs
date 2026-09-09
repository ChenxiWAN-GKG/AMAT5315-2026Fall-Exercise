// Run from the repository root with:
// cargo run --manifest-path week2/md/Cargo.toml --bin field -- week2/field.png

use md::{lennard_jones_energy, lennard_jones_force};
use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const IMAGE_WIDTH: u32 = 1120;
const IMAGE_HEIGHT: u32 = 1120;
const PLOT_LEFT: f64 = 70.0;
const PLOT_TOP: f64 = 90.0;
const PLOT_SIZE: f64 = 760.0;
const DOMAIN_RADIUS: f64 = 3.0;

fn main() -> Result<(), Box<dyn Error>> {
    let output_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("week2/field.png"));

    if let Some(parent) = output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let temporary_svg = env::temp_dir().join(format!(
        "amat5315-lennard-jones-field-{}.svg",
        std::process::id()
    ));
    fs::write(&temporary_svg, build_svg())?;

    let conversion = Command::new("/usr/bin/qlmanage")
        .args(["-t", "-s", &IMAGE_WIDTH.to_string(), "-o"])
        .arg(env::temp_dir())
        .arg(&temporary_svg)
        .output()?;
    fs::remove_file(&temporary_svg)?;

    if !conversion.status.success() {
        let message = String::from_utf8_lossy(&conversion.stderr);
        return Err(std::io::Error::other(format!(
            "Quick Look could not create the PNG: {message}"
        ))
        .into());
    }

    let generated_png = temporary_svg.with_extension("svg.png");
    fs::copy(&generated_png, &output_path)?;
    fs::remove_file(generated_png)?;

    println!("Wrote {}", output_path.display());
    Ok(())
}

fn build_svg() -> String {
    let mut svg = String::new();
    let plot_center_x = PLOT_LEFT + PLOT_SIZE / 2.0;
    let plot_center_y = PLOT_TOP + PLOT_SIZE / 2.0;
    let pixels_per_reduced_unit = PLOT_SIZE / (2.0 * DOMAIN_RADIUS);

    writeln!(
        svg,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{IMAGE_WIDTH}" height="{IMAGE_HEIGHT}" viewBox="0 0 {IMAGE_WIDTH} {IMAGE_HEIGHT}">"##
    )
    .unwrap();
    svg.push_str(
        r##"<title>Lennard-Jones pair energy and force field</title>
<desc>Pair energy is shown with blue, neutral, and red color bands. Arrows show the signed radial force around one atom.</desc>
<defs>
  <clipPath id="plot-clip"><rect x="70" y="90" width="760" height="760"/></clipPath>
  <marker id="arrowhead" markerWidth="7" markerHeight="7" refX="5.7" refY="3.5" orient="auto" markerUnits="strokeWidth">
    <path d="M 0 0 L 7 3.5 L 0 7 z" fill="#172033" stroke="#ffffff" stroke-width="0.7"/>
  </marker>
  <linearGradient id="energy-scale" x1="0" y1="1" x2="0" y2="0">
    <stop offset="0%" stop-color="#3182bd"/>
    <stop offset="50%" stop-color="#f7f7f7"/>
    <stop offset="100%" stop-color="#cb181d"/>
  </linearGradient>
</defs>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="70" y="38" font-family="Helvetica, Arial, sans-serif" font-size="25" font-weight="600" fill="#111827">Lennard–Jones pair field</text>
<text x="70" y="65" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#4b5563">Reduced units: distance r/σ, energy U/ε, force Fσ/ε</text>
<g clip-path="url(#plot-clip)">
"##,
    );

    let maximum_radius = DOMAIN_RADIUS * 2.0_f64.sqrt();
    let band_count = 900;
    for band in (1..=band_count).rev() {
        let radius = maximum_radius * f64::from(band) / f64::from(band_count);
        let pixel_radius = radius * pixels_per_reduced_unit;
        let energy = lennard_jones_energy(radius);
        let color = energy_color(energy);

        writeln!(
            svg,
            r##"  <circle cx="{plot_center_x:.2}" cy="{plot_center_y:.2}" r="{pixel_radius:.2}" fill="{color}"/>"##
        )
        .unwrap();
    }

    for tick in -3..=3 {
        let (x, _) = data_to_pixel(f64::from(tick), 0.0);
        let (_, y) = data_to_pixel(0.0, f64::from(tick));
        writeln!(
            svg,
            r##"  <line x1="{x:.2}" y1="{PLOT_TOP}" x2="{x:.2}" y2="{}" stroke="#ffffff" stroke-opacity="0.35" stroke-width="1"/>"##,
            PLOT_TOP + PLOT_SIZE
        )
        .unwrap();
        writeln!(
            svg,
            r##"  <line x1="{PLOT_LEFT}" y1="{y:.2}" x2="{}" y2="{y:.2}" stroke="#ffffff" stroke-opacity="0.35" stroke-width="1"/>"##,
            PLOT_LEFT + PLOT_SIZE
        )
        .unwrap();
    }

    for x_index in -5..=5 {
        for y_index in -5..=5 {
            let x = f64::from(x_index) * 0.5;
            let y = f64::from(y_index) * 0.5;
            let distance = x.hypot(y);

            if !(0.75..=2.8).contains(&distance) {
                continue;
            }

            let force = lennard_jones_force(distance);
            if force.abs() < 0.01 {
                continue;
            }

            append_force_arrow(&mut svg, x, y, force);
        }
    }

    let equilibrium_radius = 2.0_f64.powf(1.0 / 6.0) * pixels_per_reduced_unit;
    writeln!(
        svg,
        r##"  <circle cx="{plot_center_x:.2}" cy="{plot_center_y:.2}" r="{equilibrium_radius:.2}" fill="none" stroke="#111827" stroke-width="1.5" stroke-dasharray="6 5" opacity="0.8"/>"##
    )
    .unwrap();
    writeln!(
        svg,
        r##"  <circle cx="{plot_center_x:.2}" cy="{plot_center_y:.2}" r="16" fill="#111827" stroke="#ffffff" stroke-width="3"/>"##
    )
    .unwrap();
    svg.push_str("</g>\n");

    writeln!(
        svg,
        r##"<rect x="{PLOT_LEFT}" y="{PLOT_TOP}" width="{PLOT_SIZE}" height="{PLOT_SIZE}" fill="none" stroke="#111827" stroke-width="1.5"/>"##
    )
    .unwrap();

    for tick in -3..=3 {
        let (x, _) = data_to_pixel(f64::from(tick), 0.0);
        let (_, y) = data_to_pixel(0.0, f64::from(tick));
        writeln!(
            svg,
            r##"<line x1="{x:.2}" y1="850" x2="{x:.2}" y2="856" stroke="#111827"/><text x="{x:.2}" y="875" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#374151">{tick}</text>"##
        )
        .unwrap();
        writeln!(
            svg,
            r##"<line x1="64" y1="{y:.2}" x2="70" y2="{y:.2}" stroke="#111827"/><text x="58" y="{:.2}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#374151">{tick}</text>"##,
            y + 4.5
        )
        .unwrap();
    }

    svg.push_str(
        r##"<text x="450" y="906" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#111827">x/σ</text>
<text x="18" y="470" text-anchor="middle" transform="rotate(-90 18 470)" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#111827">y/σ</text>
<rect x="870" y="155" width="28" height="330" fill="url(#energy-scale)" stroke="#111827" stroke-width="1"/>
<text x="870" y="128" font-family="Helvetica, Arial, sans-serif" font-size="16" font-weight="600" fill="#111827">Energy U/ε</text>
<text x="910" y="164" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#374151">+1 (clipped)</text>
<text x="910" y="325" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#374151">0</text>
<text x="910" y="485" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#374151">−1</text>
<text x="870" y="555" font-family="Helvetica, Arial, sans-serif" font-size="16" font-weight="600" fill="#111827">Force arrows</text>
<line x1="872" y1="590" x2="938" y2="590" stroke="#ffffff" stroke-width="5"/>
<line x1="872" y1="590" x2="938" y2="590" stroke="#172033" stroke-width="2" marker-end="url(#arrowhead)"/>
<text x="870" y="620" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#374151">Direction is exact.</text>
<text x="870" y="642" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#374151">Length is compressed</text>
<text x="870" y="661" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#374151">nonlinearly.</text>
<circle cx="883" cy="718" r="10" fill="#111827" stroke="#ffffff" stroke-width="2"/>
<text x="904" y="723" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#374151">fixed atom</text>
<line x1="871" y1="770" x2="899" y2="770" stroke="#111827" stroke-width="1.5" stroke-dasharray="6 5"/>
<text x="870" y="798" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#374151">F = 0 at</text>
<text x="870" y="818" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#374151">r = 2¹ᐟ⁶σ</text>
</svg>
"##,
    );

    svg
}

fn data_to_pixel(x: f64, y: f64) -> (f64, f64) {
    let scale = PLOT_SIZE / (2.0 * DOMAIN_RADIUS);
    let pixel_x = PLOT_LEFT + (x + DOMAIN_RADIUS) * scale;
    let pixel_y = PLOT_TOP + (DOMAIN_RADIUS - y) * scale;
    (pixel_x, pixel_y)
}

fn append_force_arrow(svg: &mut String, x: f64, y: f64, force: f64) {
    let distance = x.hypot(y);
    let direction = force.signum();
    let arrow_length = 30.0 * (force.abs() / (force.abs() + 0.5)).sqrt();
    let pixel_dx = direction * x / distance * arrow_length;
    let pixel_dy = -direction * y / distance * arrow_length;
    let (center_x, center_y) = data_to_pixel(x, y);
    let start_x = center_x - pixel_dx / 2.0;
    let start_y = center_y - pixel_dy / 2.0;
    let end_x = center_x + pixel_dx / 2.0;
    let end_y = center_y + pixel_dy / 2.0;

    writeln!(
        svg,
        r##"  <line x1="{start_x:.2}" y1="{start_y:.2}" x2="{end_x:.2}" y2="{end_y:.2}" stroke="#ffffff" stroke-width="5" stroke-linecap="round" opacity="0.9"/>"##
    )
    .unwrap();
    writeln!(
        svg,
        r##"  <line x1="{start_x:.2}" y1="{start_y:.2}" x2="{end_x:.2}" y2="{end_y:.2}" stroke="#172033" stroke-width="2" stroke-linecap="round" marker-end="url(#arrowhead)"/>"##
    )
    .unwrap();
}

fn energy_color(energy: f64) -> String {
    const ATTRACTIVE: [u8; 3] = [49, 130, 189];
    const NEUTRAL: [u8; 3] = [247, 247, 247];
    const REPULSIVE: [u8; 3] = [203, 24, 29];

    let clipped_energy = energy.clamp(-1.0, 1.0);
    let color = if clipped_energy < 0.0 {
        mix_colors(ATTRACTIVE, NEUTRAL, clipped_energy + 1.0)
    } else {
        mix_colors(NEUTRAL, REPULSIVE, clipped_energy)
    };

    format!("#{:02x}{:02x}{:02x}", color[0], color[1], color[2])
}

fn mix_colors(start: [u8; 3], end: [u8; 3], fraction: f64) -> [u8; 3] {
    let mix_channel = |start: u8, end: u8| {
        (f64::from(start) + fraction * (f64::from(end) - f64::from(start))).round() as u8
    };

    [
        mix_channel(start[0], end[0]),
        mix_channel(start[1], end[1]),
        mix_channel(start[2], end[2]),
    ]
}
