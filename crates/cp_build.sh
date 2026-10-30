#!/usr/bin/env bash
set -euo pipefail
profile=${1:-release}
destination=${2:-"$HOME/tmp/thoosa_build"}
[[ "$profile" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid build profile' >&2; exit 1; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir="$script_dir/target/${CARGO_BUILD_TARGET:-x86_64-unknown-linux-gnu}/$profile"
# Validate inputs before touching the destination, then publish a complete directory.
[[ -f "$source_dir/thoosa_hbac" && -f "$source_dir/libthoosa_shims.so" ]] || { echo "Missing build artifacts in $source_dir" >&2; exit 1; }
mkdir -p -- "$(dirname -- "$destination")"
[[ ! -L "$destination" ]] || { echo "Destination must not be a symlink" >&2; exit 1; }
destination=$(realpath -m -- "$destination")
[[ "$destination" != / && "$destination" != "$HOME" && "$source_dir/" != "$destination/"* ]] || { echo 'Invalid destination' >&2; exit 1; }
staging=$(mktemp -d "${destination}.staging.XXXXXX")
backup="${staging}.previous"
cleanup() {
    rm -rf -- "$staging"
    if [[ -e "$backup" && ! -e "$destination" ]]; then mv -- "$backup" "$destination"; fi
}
trap cleanup EXIT
for artifact in "$source_dir"/thoosa_* "$source_dir"/libthoosa*.so; do
    [[ -f "$artifact" ]] && cp -- "$artifact" "$staging/"
done
if [[ -e "$destination" ]]; then mv -- "$destination" "$backup"; fi
mv -- "$staging" "$destination"
rm -rf -- "$backup"
