if [ "$1" == "slow" ]; then
    export LD_LIBRARY_PATH=/home/bu/dvl/torch/libtorch-2.7.1/lib/ && export RUST_LOG=Info && CARGO_TARGET_DIR=target && cargo nextest run -r --no-fail-fast -E 'test(/slow*/)'
else
    export LD_LIBRARY_PATH=/home/bu/dvl/torch/libtorch-2.7.1/lib/ && export RUST_LOG=Info && CARGO_TARGET_DIR=target && cargo nextest run -r --no-fail-fast --retries 0 -E 'not (test(/slow*/))'
fi
#single
#--package thoosa_common --lib argmin::tests::test_min_rbf