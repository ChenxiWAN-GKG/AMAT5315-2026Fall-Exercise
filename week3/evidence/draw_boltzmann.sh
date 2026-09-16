#!/bin/sh
set -eu

exec python3 "$(dirname "$0")/plot_boltzmann.py" "$@"
