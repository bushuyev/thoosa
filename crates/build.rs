use std::fmt::Debug;

use chrono::prelude::*;

fn main() {
    let libtorch = std::env::var("LIBTORCH")
        .expect("LIBTORCH must be set (see .cargo/config.toml)");
    let libtorch = libtorch.trim_end_matches('/');
    println!("cargo:rustc-env=LD_LIBRARY_PATH={libtorch}/lib");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rustc-env=CARGO_TARGET_DIR=target");//https://stackoverflow.com/a/73543282/656449 - avoid rebuild all deps
}
