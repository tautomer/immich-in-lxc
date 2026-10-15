#!/bin/bash

# Print the ML interpreter version for an Immich release tag.
set -euo pipefail

version=${1:-}
if [[ ! "$version" =~ ^v?([0-9]+)\.([0-9]+)\.([0-9]+)([-+].*)?$ ]]; then
    echo "ERROR: Invalid Immich release tag: $version" >&2
    exit 1
fi
major=$((10#${BASH_REMATCH[1]}))
minor=$((10#${BASH_REMATCH[2]}))

if (( major > 3 || (major == 3 && minor >= 3) )); then
    echo 3.12
else
    echo 3.11
fi
