
fn main() {
    println!("cargo:rustc-env=LD_LIBRARY_PATH=/home/bu/dvl/torch/src/pytorch-dst/lib/");
    println!("cargo:rustc-env=CARGO_TARGET_DIR=target");//https://stackoverflow.com/a/73543282/656449 - avoid rebuild all deps
}
