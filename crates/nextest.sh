if [ "$1" == "slow" ]; then
    export LD_LIBRARY_PATH=/home/bu/dvl/torch/src/pytorch-dst/lib/ && CARGO_TARGET_DIR=target && cargo nextest run --no-fail-fast -E 'test(/slow*/)'
else
    export LD_LIBRARY_PATH=/home/bu/dvl/torch/src/pytorch-dst/lib/ && CARGO_TARGET_DIR=target && cargo nextest run --no-fail-fast --retries 2 -E 'not (test(/slow*/))'
fi