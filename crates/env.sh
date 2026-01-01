if [ -z "${LIBTORCH:-}" ]; then
    LIBTORCH=$(awk -F'"' '/^LIBTORCH=/{print $2}' "$(dirname "$0")/.cargo/config.toml")
fi
LIBTORCH="${LIBTORCH%/}"

#LD_PRELOAD="${LIBTORCH}/lib/libtorch_cuda.so:${LIBTORCH}/lib/libtorch_cpu.so:${LIBTORCH}/lib/libc10.so"
export LD_PRELOAD="${LIBTORCH}/lib/libtorch_cuda.so:${LIBTORCH}/lib/libtorch_cpu.so:${LIBTORCH}/lib/libc10.so"
