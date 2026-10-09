"""Require every README evidence item to be tracked and non-empty."""

from pathlib import Path
import subprocess


week = Path(__file__).resolve().parents[1]
inventory = (week / "README.md").read_text().split("## Evidence inventory\n", 1)[1].split("\n## ", 1)[0]
names = []
for line in inventory.splitlines():
    if not line.startswith("| `"):
        continue
    first_column = line.split("|", 2)[1]
    names.extend(part.strip().strip("` ") for part in first_column.split(","))

for name in names:
    path = week / name
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"evidence missing or empty: {name}")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", f"week2/{name}"],
        cwd=week.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if tracked.returncode:
        raise SystemExit(f"evidence not tracked: {name}")
print(f"All {len(names)} README evidence files are tracked and non-empty.")
