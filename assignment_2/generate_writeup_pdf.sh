#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_dir="$script_dir/output"
output_file="$output_dir/assignment_2_WRITEUP.pdf"

command -v pandoc >/dev/null 2>&1 || {
    echo "Error: pandoc is required to generate the PDF." >&2
    exit 1
}

command -v xelatex >/dev/null 2>&1 || {
    echo "Error: xelatex is required to generate the PDF." >&2
    exit 1
}

mkdir -p "$output_dir"

cd "$script_dir"
pandoc assignment_2_WRITEUP.md \
    --from=markdown \
    --pdf-engine=xelatex \
    --lua-filter="$script_dir/writeup_filter.lua" \
    --include-in-header="$script_dir/writeup_header.tex" \
    --resource-path="$script_dir" \
    --variable=geometry:margin=1in \
    --output="$output_file"

echo "Generated $output_file"
