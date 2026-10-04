#!/usr/bin/env bash
# Usage: bash sync.sh host:/absolute/path/to/KernelLab
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[^/:[:space:]]+:.+ ]]; then
  printf 'Usage: bash %s host:/path\n' "${0##*/}" >&2
  exit 2
fi

if ! command -v rsync >/dev/null 2>&1; then
  printf 'Error: rsync is required. Run this script in Linux or WSL with rsync installed.\n' >&2
  exit 1
fi

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
destination="${1%/}/"

# The trailing slash copies project contents into the destination directory.
# Keep remote build outputs and other remote-only files intact (no --delete).
rsync -avz --protect-args -e ssh \
  --exclude='.git/' \
  --exclude='build/' \
  --exclude='Build/' \
  --exclude='build-*/' \
  --exclude='CMakeFiles/' \
  --exclude='CMakeCache.txt' \
  --exclude='cmake_install.cmake' \
  --exclude='install_manifest.txt' \
  --exclude='compile_commands.json' \
  --exclude='Testing/' \
  -- "$project_dir/" "$destination"
