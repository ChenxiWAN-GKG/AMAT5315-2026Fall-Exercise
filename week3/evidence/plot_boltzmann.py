#!/usr/bin/env python3
"""Draw the two-temperature Boltzmann histogram and log-ratio check."""

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape


BIN_WIDTH = 40
MIN_COUNT = 5
TEMPERATURES = (3.0, 3.1)
WIDTH = 1200
HEIGHT = 1200
LEFT = 90
RIGHT = 1120
HIST_TOP = 90
HIST_BOTTOM = 370
RATIO_TOP = 470
RATIO_BOTTOM = 750


def read_total_energies(path: Path) -> list[float]:
    energies = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            try:
                side = int(record["L"])
                energy_per_site = float(record["E"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid energy record at {path}:{line_number}") from error
            energies.append(energy_per_site * side * side)
    if not energies:
        raise ValueError(f"no rows found in {path}")
    return energies


def histogram(values: list[float], low: float, bin_count: int) -> list[int]:
    counts = [0] * bin_count
    for value in values:
        index = math.floor((value - low) / BIN_WIDTH)
        if index == bin_count:
            index -= 1
        if not 0 <= index < bin_count:
            raise ValueError(f"energy {value} falls outside the shared histogram bins")
        counts[index] += 1
    return counts


def nice_step(raw_step: float) -> float:
    if raw_step <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    scale = 10**exponent
    normalized = raw_step / scale
    if normalized <= 1:
        multiplier = 1
    elif normalized <= 2:
        multiplier = 2
    elif normalized <= 5:
        multiplier = 5
    else:
        multiplier = 10
    return multiplier * scale


def ticks(low: float, high: float, count: int = 5) -> list[float]:
    step = nice_step((high - low) / count)
    first = math.ceil(low / step) * step
    values = []
    value = first
    while value <= high + step * 1e-9:
        values.append(value)
        value += step
    return values


def sx(value: float, low: float, high: float) -> float:
    return LEFT + (value - low) / (high - low) * (RIGHT - LEFT)


def sy(value: float, low: float, high: float, top: float, bottom: float) -> float:
    return bottom - (value - low) / (high - low) * (bottom - top)


def text(x: float, y: float, value: str, size: int = 16, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}px" '
        f'text-anchor="{anchor}">{escape(value)}</text>'
    )


def draw_axes(
    parts: list[str],
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
    top: float,
    bottom: float,
    y_label: str,
) -> None:
    for value in ticks(y_low, y_high):
        y = sy(value, y_low, y_high, top, bottom)
        parts.append(f'<line x1="{LEFT}" y1="{y:.2f}" x2="{RIGHT}" y2="{y:.2f}" class="grid"/>')
        parts.append(text(LEFT - 12, y + 5, f"{value:g}", 14, "end"))
    parts.append(f'<line x1="{LEFT}" y1="{bottom}" x2="{RIGHT}" y2="{bottom}" class="axis"/>')
    parts.append(f'<line x1="{LEFT}" y1="{top}" x2="{LEFT}" y2="{bottom}" class="axis"/>')
    for value in range(math.ceil(x_low / 80) * 80, math.floor(x_high / 80) * 80 + 1, 80):
        x = sx(value, x_low, x_high)
        parts.append(f'<line x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" y2="{bottom + 7}" class="axis"/>')
        parts.append(text(x, bottom + 28, f"{value:g}", 14))
    parts.append(
        f'<text x="18" y="{(top + bottom) / 2:.2f}" font-size="16px" '
        f'text-anchor="middle" transform="rotate(-90 18 {(top + bottom) / 2:.2f})">{escape(y_label)}</text>'
    )


def make_svg(
    counts_30: list[int],
    counts_31: list[int],
    x_low: float,
    x_high: float,
    valid_centers: list[float],
    log_ratios: list[float],
    expected_slope: float,
) -> str:
    hist_high = max(max(counts_30), max(counts_31)) * 1.1
    ratio_line_intercept = sum(
        ratio - expected_slope * center for center, ratio in zip(valid_centers, log_ratios)
    ) / len(log_ratios)
    line_values = [ratio_line_intercept + expected_slope * center for center in valid_centers]
    ratio_low = min(min(log_ratios), min(line_values))
    ratio_high = max(max(log_ratios), max(line_values))
    ratio_padding = max(0.05, 0.12 * (ratio_high - ratio_low))
    ratio_low -= ratio_padding
    ratio_high += ratio_padding

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<style>"
        ".axis{stroke:#111827;stroke-width:1.5;fill:none;}"
        ".grid{stroke:#d1d5db;stroke-width:1;fill:none;}"
        ".label{fill:#111827;font-family:Arial,sans-serif;}"
        ".blue{fill:#2563eb;fill-opacity:.28;stroke:#2563eb;stroke-width:1.2;}"
        ".orange{fill:#d97706;fill-opacity:.28;stroke:#d97706;stroke-width:1.2;}"
        ".expected{stroke:#111827;stroke-width:2;stroke-dasharray:8 6;fill:none;}"
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g class="label">',
        text(WIDTH / 2, 34, "2D Ising Boltzmann check", 24),
        text(WIDTH / 2, 61, "Total energy E × 4096; shared bins are 40 units wide", 16),
    ]

    draw_axes(parts, x_low, x_high, 0, hist_high, HIST_TOP, HIST_BOTTOM, "rows per bin")
    bin_count = len(counts_30)
    bar_width = (RIGHT - LEFT) / bin_count
    for index, count in enumerate(counts_30):
        x = LEFT + index * bar_width + 1
        y = sy(count, 0, hist_high, HIST_TOP, HIST_BOTTOM)
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 2:.2f}" '
            f'height="{HIST_BOTTOM - y:.2f}" class="blue"/>'
        )
    for index, count in enumerate(counts_31):
        x = LEFT + index * bar_width + 1
        y = sy(count, 0, hist_high, HIST_TOP, HIST_BOTTOM)
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 2:.2f}" '
            f'height="{HIST_BOTTOM - y:.2f}" class="orange"/>'
        )
    parts.extend(
        [
            '<rect x="910" y="112" width="18" height="18" class="blue"/>',
            text(938, 127, "T = 3.0", 16, "start"),
            '<rect x="910" y="143" width="18" height="18" class="orange"/>',
            text(938, 158, "T = 3.1", 16, "start"),
        ]
    )

    draw_axes(parts, x_low, x_high, ratio_low, ratio_high, RATIO_TOP, RATIO_BOTTOM, "log(N₃.₁ / N₃.₀)")
    for center, ratio in zip(valid_centers, log_ratios):
        x = sx(center, x_low, x_high)
        y = sy(ratio, ratio_low, ratio_high, RATIO_TOP, RATIO_BOTTOM)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#2563eb"/>')
    line_x_low = min(valid_centers)
    line_x_high = max(valid_centers)
    parts.append(
        f'<line x1="{sx(line_x_low, x_low, x_high):.2f}" '
        f'y1="{sy(ratio_line_intercept + expected_slope * line_x_low, ratio_low, ratio_high, RATIO_TOP, RATIO_BOTTOM):.2f}" '
        f'x2="{sx(line_x_high, x_low, x_high):.2f}" '
        f'y2="{sy(ratio_line_intercept + expected_slope * line_x_high, ratio_low, ratio_high, RATIO_TOP, RATIO_BOTTOM):.2f}" '
        'class="expected"/>'
    )
    parts.extend(
        [
            text(RIGHT, RATIO_TOP + 27, f"slope = {expected_slope:.6f}", 16, "end"),
            text(RIGHT, RATIO_TOP + 51, f"{len(valid_centers)} bins with N ≥ {MIN_COUNT} in both", 16, "end"),
            '<line x1="910" y1="558" x2="948" y2="558" class="expected"/>',
            text(956, 564, "fixed-slope overlay", 16, "start"),
            text(WIDTH / 2, 820, "total energy", 16),
            text(WIDTH / 2, 850, "Only bins with N ≥ 5 in both runs enter the ratio.", 14),
            "</g>",
            "</svg>",
        ]
    )
    return "\n".join(parts)


def render_png(svg: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        svg_path = Path(directory) / "boltzmann.svg"
        svg_path.write_text(svg, encoding="utf-8")
        subprocess.run(
            ["qlmanage", "-t", "-s", str(WIDTH), "-o", directory, str(svg_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        thumbnail = svg_path.with_name("boltzmann.svg.png")
        if not thumbnail.exists():
            raise RuntimeError("qlmanage did not produce a PNG preview")
        shutil.copyfile(thumbnail, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "runs",
        help="directory containing T3.0/ and T3.1/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "boltzmann.png",
        help="PNG output path",
    )
    args = parser.parse_args()

    energies = [
        read_total_energies(args.data_root / "T3.0" / "series.jsonl"),
        read_total_energies(args.data_root / "T3.1" / "series.jsonl"),
    ]
    low = BIN_WIDTH * math.floor(min(min(values) for values in energies) / BIN_WIDTH)
    high = BIN_WIDTH * math.ceil(max(max(values) for values in energies) / BIN_WIDTH)
    if high == low:
        high += BIN_WIDTH
    if max(max(values) for values in energies) == high:
        high += BIN_WIDTH
    bin_count = int((high - low) / BIN_WIDTH)
    counts_30 = histogram(energies[0], low, bin_count)
    counts_31 = histogram(energies[1], low, bin_count)
    valid = [
        index for index, (count_30, count_31) in enumerate(zip(counts_30, counts_31))
        if count_30 >= MIN_COUNT and count_31 >= MIN_COUNT
    ]
    if not valid:
        raise ValueError("no shared bins meet the minimum count")
    centers = [low + (index + 0.5) * BIN_WIDTH for index in valid]
    ratios = [math.log(counts_31[index] / counts_30[index]) for index in valid]
    slope = 1 / TEMPERATURES[0] - 1 / TEMPERATURES[1]
    svg = make_svg(counts_30, counts_31, low, high, centers, ratios, slope)
    render_png(svg, args.out)
    print(f"wrote {args.out} ({len(energies[0])} rows at T=3.0, {len(energies[1])} rows at T=3.1)")
    print(f"shared bins: {len(valid)}; expected slope: {slope:.12f}")


if __name__ == "__main__":
    main()
