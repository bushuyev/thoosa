if [ -z "${LIBTORCH:-}" ]; then
    LIBTORCH=$(awk -F'"' '/^LIBTORCH=/{print $2}' "$(dirname "$0")/.cargo/config.toml")
fi
LIBTORCH="${LIBTORCH%/}"

if [ "$1" == "slow" ]; then
    export LD_LIBRARY_PATH="${LIBTORCH}/lib" && export RUST_LOG=Info && CARGO_TARGET_DIR=target && cargo nextest run -r --no-fail-fast -E 'test(/slow*/)'
else
    export LD_LIBRARY_PATH="${LIBTORCH}/lib" && export RUST_LOG=Info && CARGO_TARGET_DIR=target && cargo nextest run -r --no-fail-fast --retries 0 -E 'not (test(/slow*/))'
fi
#single
#--package thoosa_common --lib argmin::tests::test_min_rbf
