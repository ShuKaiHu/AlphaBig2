#!/usr/bin/env bash

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
venv_activate="$repo_root/.venv/bin/activate"

if [[ ! -f "$venv_activate" ]]; then
  echo "Missing virtualenv: $venv_activate"
  exit 1
fi

# shellcheck source=/dev/null
source "$venv_activate"

# Keep Python resolution isolated from any user/global site-packages.
export PYTHONNOUSERSITE=1
hash -r
