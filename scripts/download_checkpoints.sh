#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <url> <sha256> <output_dir> [output_name]" >&2
  exit 2
fi

url=$1
expected_sha256=${2,,}
output_dir=$3
output_name=${4:-${url%%\?*}}
output_name=${output_name##*/}

if [[ ! $expected_sha256 =~ ^[0-9a-f]{64}$ ]]; then
  echo "Error: sha256 must be 64 hexadecimal characters." >&2
  exit 2
fi
if [[ -z $output_name || $output_name == . || $output_name == .. || $output_name == */* || $output_name == *\\* ]]; then
  echo "Error: invalid output filename." >&2
  exit 2
fi

mkdir -p -- "$output_dir"
tmp_file=$(mktemp "$output_dir/.download.XXXXXX")
trap 'rm -f -- "$tmp_file"' EXIT
curl --fail --location --retry 3 --output "$tmp_file" "$url"
printf '%s  %s\n' "$expected_sha256" "$tmp_file" | sha256sum --check --status
mv -- "$tmp_file" "$output_dir/$output_name"
trap - EXIT
printf 'Saved: %s\n' "$output_dir/$output_name"
