use std::fs;
use chrono::Utc;

fn main() {
    println!("cargo:rustc-env=LD_LIBRARY_PATH=/home/bu/dvl/torch/libtorch-2.5.0/lib/");
    println!("cargo:rustc-env=CARGO_TARGET_DIR=target");//https://stackoverflow.com/a/73543282/656449 - avoid rebuild all deps
    // fs::write("src/version.rs", format!("pub const VERSION:&'static str = \"{}\";", Utc::now())).expect("Unable to write file");
}
