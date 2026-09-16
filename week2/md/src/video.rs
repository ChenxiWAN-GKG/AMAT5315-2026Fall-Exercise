use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use super::io::{RunMetadata, TrajectoryFrame, read_run_metadata, read_trajectory};
use super::rdf::{RdfBin, rdf_for_frames};

pub const VIDEO_WIDTH: usize = 640;
pub const VIDEO_HEIGHT: usize = 360;
pub const RDF_BINS: usize = 50;
const VIDEO_FRAME_RATE: usize = 20;
const MAX_VIDEO_BYTES: u64 = 2_000_000;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VideoError {
    EncoderUnavailable,
    EncodingOrSize,
}

impl fmt::Display for VideoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EncoderUnavailable => {
                formatter.write_str("ffmpeg encoder unavailable; install ffmpeg and retry")
            }
            Self::EncodingOrSize => {
                formatter.write_str("video encoding failed or output exceeded 2,000,000 bytes")
            }
        }
    }
}

impl fmt::Debug for VideoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_string())
    }
}

impl Error for VideoError {}

pub fn render_video(input: &Path, output: &Path) -> Result<()> {
    let metadata = read_run_metadata(input)?;
    let frames = read_trajectory(input)?;
    if frames.is_empty() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidData,
            "trajectory contains no frames",
        )));
    }

    let video_size = format!("{}x{}", VIDEO_WIDTH, VIDEO_HEIGHT);
    let frame_rate = VIDEO_FRAME_RATE.to_string();
    let mut encoder = match Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            &video_size,
            "-framerate",
            &frame_rate,
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "35",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(output)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Err(Box::new(VideoError::EncoderUnavailable));
        }
        Err(error) => return Err(Box::new(error)),
    };

    let mut encoder_stdin = match encoder.stdin.take() {
        Some(stdin) => stdin,
        None => {
            let _ = encoder.wait();
            return Err(Box::new(VideoError::EncoderUnavailable));
        }
    };

    for (frame_index, frame) in frames.iter().enumerate() {
        let rdf = rdf_for_frames(&metadata, &frames, frame_index, RDF_BINS);
        let rgb = render_rgb_frame(
            &metadata,
            frame,
            &rdf,
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
        );
        if encoder_stdin.write_all(&rgb).is_err() {
            drop(encoder_stdin);
            let _ = encoder.wait();
            return Err(Box::new(VideoError::EncodingOrSize));
        }
    }
    drop(encoder_stdin);

    let status = encoder.wait()?;
    if !status.success() {
        return Err(Box::new(VideoError::EncodingOrSize));
    }
    let size = fs::metadata(output)?.len();
    if size >= MAX_VIDEO_BYTES {
        return Err(Box::new(VideoError::EncodingOrSize));
    }
    Ok(())
}

pub fn render_rgb_frame(
    metadata: &RunMetadata,
    frame: &TrajectoryFrame,
    rdf: &[RdfBin],
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut rgb = vec![24_u8; width.saturating_mul(height).saturating_mul(3)];
    if width == 0 || height == 0 {
        return rgb;
    }

    let left_width = width / 2;
    let margin = 18_i32;
    let left_x0 = margin;
    let left_x1 = (left_width as i32 - margin - 1).max(left_x0);
    let panel_y0 = margin;
    let panel_y1 = (height as i32 - margin - 1).max(panel_y0);
    let right_x0 = (left_width as i32 + margin).min(width as i32 - 1);
    let right_x1 = (width as i32 - margin - 1).max(right_x0);

    fill_rect(
        &mut rgb,
        width,
        left_x0,
        panel_y0,
        left_x1,
        panel_y1,
        [28, 34, 44],
    );
    fill_rect(
        &mut rgb,
        width,
        right_x0,
        panel_y0,
        right_x1,
        panel_y1,
        [28, 34, 44],
    );
    draw_rect(
        &mut rgb,
        width,
        left_x0,
        panel_y0,
        left_x1,
        panel_y1,
        [150, 160, 175],
    );
    draw_rect(
        &mut rgb,
        width,
        right_x0,
        panel_y0,
        right_x1,
        panel_y1,
        [150, 160, 175],
    );

    draw_atoms(
        &mut rgb,
        width,
        metadata,
        frame,
        left_x0 + 1,
        panel_y0 + 1,
        (left_x1 - left_x0 - 1).max(1),
        (panel_y1 - panel_y0 - 1).max(1),
    );
    draw_rdf(
        &mut rgb,
        width,
        metadata,
        rdf,
        right_x0,
        panel_y0,
        right_x1,
        panel_y1,
    );
    rgb
}

fn draw_atoms(
    rgb: &mut [u8],
    width: usize,
    metadata: &RunMetadata,
    frame: &TrajectoryFrame,
    x0: i32,
    y0: i32,
    plot_width: i32,
    plot_height: i32,
) {
    let [lx, ly] = metadata.box_size;
    if !(lx.is_finite() && ly.is_finite() && lx > 0.0 && ly > 0.0) {
        return;
    }
    for position in frame.pos.iter().take(metadata.n) {
        if !(position[0].is_finite() && position[1].is_finite()) {
            continue;
        }
        let x = x0 + (position[0].rem_euclid(lx) / lx * plot_width as f64) as i32;
        let y = y0 + (position[1].rem_euclid(ly) / ly * plot_height as f64) as i32;
        draw_circle(rgb, width, x, y, 4, [245, 150, 45]);
    }
}

fn draw_rdf(
    rgb: &mut [u8],
    width: usize,
    metadata: &RunMetadata,
    rdf: &[RdfBin],
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
) {
    let axis_x0 = x0 + 12;
    let axis_x1 = (x1 - 8).max(axis_x0);
    let axis_y0 = (y1 - 12).max(y0);
    let axis_y1 = y0 + 8;
    draw_line(rgb, width, axis_x0, axis_y0, axis_x1, axis_y0, [180, 190, 205]);
    draw_line(rgb, width, axis_x0, axis_y0, axis_x0, axis_y1, [180, 190, 205]);
    if rdf.is_empty() {
        return;
    }

    let rmax = metadata.box_size[0].min(metadata.box_size[1]) / 2.0;
    if !(rmax.is_finite() && rmax > 0.0) {
        return;
    }
    let max_g = rdf
        .iter()
        .map(|bin| bin.g)
        .filter(|value| value.is_finite() && *value > 0.0)
        .fold(1.0_f64, f64::max);
    let mut previous = None;
    for bin in rdf {
        if !(bin.r.is_finite() && bin.g.is_finite()) {
            previous = None;
            continue;
        }
        let x_ratio = (bin.r / rmax).clamp(0.0, 1.0);
        let y_ratio = (bin.g / max_g).clamp(0.0, 1.0);
        let x = axis_x0 + (x_ratio * (axis_x1 - axis_x0) as f64) as i32;
        let y = axis_y0 - (y_ratio * (axis_y0 - axis_y1) as f64) as i32;
        if let Some((previous_x, previous_y)) = previous {
            draw_line(
                rgb,
                width,
                previous_x,
                previous_y,
                x,
                y,
                [70, 165, 245],
            );
        }
        draw_circle(rgb, width, x, y, 2, [70, 165, 245]);
        previous = Some((x, y));
    }
}

fn fill_rect(
    rgb: &mut [u8],
    width: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: [u8; 3],
) {
    for y in y0..=y1 {
        for x in x0..=x1 {
            set_pixel(rgb, width, x, y, color);
        }
    }
}

fn draw_rect(
    rgb: &mut [u8],
    width: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: [u8; 3],
) {
    draw_line(rgb, width, x0, y0, x1, y0, color);
    draw_line(rgb, width, x1, y0, x1, y1, color);
    draw_line(rgb, width, x1, y1, x0, y1, color);
    draw_line(rgb, width, x0, y1, x0, y0, color);
}

fn draw_circle(rgb: &mut [u8], width: usize, center_x: i32, center_y: i32, radius: i32, color: [u8; 3]) {
    for y in -radius..=radius {
        for x in -radius..=radius {
            if x * x + y * y <= radius * radius {
                set_pixel(rgb, width, center_x + x, center_y + y, color);
            }
        }
    }
}

fn draw_line(
    rgb: &mut [u8],
    width: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: [u8; 3],
) {
    let steps = (x1 - x0).abs().max((y1 - y0).abs());
    if steps == 0 {
        set_pixel(rgb, width, x0, y0, color);
        return;
    }
    for step in 0..=steps {
        let fraction = step as f64 / steps as f64;
        let x = x0 as f64 + (x1 - x0) as f64 * fraction;
        let y = y0 as f64 + (y1 - y0) as f64 * fraction;
        set_pixel(rgb, width, x.round() as i32, y.round() as i32, color);
    }
}

fn set_pixel(rgb: &mut [u8], width: usize, x: i32, y: i32, color: [u8; 3]) {
    if x < 0 || y < 0 || width == 0 {
        return;
    }
    let x = x as usize;
    let y = y as usize;
    let height = rgb.len() / (width * 3);
    if x >= width || y >= height {
        return;
    }
    let index = (y * width + x) * 3;
    rgb[index..index + 3].copy_from_slice(&color);
}
