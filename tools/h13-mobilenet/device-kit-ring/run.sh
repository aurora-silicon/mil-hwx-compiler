#!/bin/bash
set -euo pipefail

# Run on ANECOMP's M1 Pro with libane's anechain (which times ane_exec).
# usage: ANECHAIN=/path/to/anechain TOOLS=/path/to/tools/h13-mobilenet ./run.sh [repeat]
kit=$(cd "$(dirname "$0")" && pwd)
repeat=${1:-100}
tools=${TOOLS:-$(cd "$kit/../tools/h13-mobilenet" 2>/dev/null && pwd || true)}
: "${tools:?set TOOLS to tools/h13-mobilenet}"
: "${ANECHAIN:=$tools/anechain}"
export ANECHAIN

for name in stem-k3-s2-224 dw-k3-s2-112 dw-k3-s2-56 dw-k3-s2-28 dw-k3-s2-14; do
    case="$kit/singles/$name"
    python3 "$tools/run_chain.py" "$case" "$repeat" --no-perop
done
"$tools/run_prefixes.sh" "$kit/prefixes" "$repeat"
python3 "$tools/run_chain.py" "$kit/full" "$repeat" --no-perop
