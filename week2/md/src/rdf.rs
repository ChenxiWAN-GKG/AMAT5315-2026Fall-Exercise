use super::io::{RunMetadata, TrajectoryFrame};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RdfBin {
    pub r: f64,
    pub g: f64,
}

pub fn rdf_for_frames(
    metadata: &RunMetadata,
    frames: &[TrajectoryFrame],
    through_frame: usize,
    bin_count: usize,
) -> Vec<RdfBin> {
    if bin_count == 0 {
        return Vec::new();
    }

    let rmax = metadata.box_size[0].min(metadata.box_size[1]) / 2.0;
    if !(rmax.is_finite() && rmax > 0.0) {
        return vec![RdfBin { r: 0.0, g: 0.0 }; bin_count];
    }
    let dr = rmax / bin_count as f64;
    let mut result = (0..bin_count)
        .map(|bin| RdfBin {
            r: (bin as f64 + 0.5) * dr,
            g: 0.0,
        })
        .collect::<Vec<_>>();

    let frame_count = frames.len().min(through_frame.saturating_add(1));
    if frame_count == 0 || metadata.n == 0 {
        return result;
    }

    let mut neighbour_counts = vec![0.0; bin_count];
    for frame in frames.iter().take(frame_count) {
        for i in 0..metadata.n {
            for j in 0..metadata.n {
                if i == j {
                    continue;
                }
                let distance = minimum_image_distance(metadata, frame, i, j);
                if distance < rmax {
                    let bin = (distance / dr) as usize;
                    if bin < bin_count {
                        neighbour_counts[bin] += 1.0;
                    }
                }
            }
        }
    }

    let samples = (frame_count * metadata.n) as f64;
    for bin in 0..bin_count {
        let r0 = bin as f64 * dr;
        let r1 = r0 + dr;
        let measured = neighbour_counts[bin] / samples;
        let expected = metadata.rho * std::f64::consts::PI * (r1 * r1 - r0 * r0);
        result[bin].g = measured / expected;
    }
    result
}

fn minimum_image_distance(
    metadata: &RunMetadata,
    frame: &TrajectoryFrame,
    i: usize,
    j: usize,
) -> f64 {
    let [lx, ly] = metadata.box_size;
    let mut dx = frame.pos[i][0] - frame.pos[j][0];
    let mut dy = frame.pos[i][1] - frame.pos[j][1];
    dx -= lx * (dx / lx).round();
    dy -= ly * (dy / ly).round();
    dx.hypot(dy)
}
