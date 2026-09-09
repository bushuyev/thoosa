use std::fs;
use chrono::Utc;

fn main() {
    let libtorch = std::env::var("LIBTORCH")
        .expect("LIBTORCH must be set (see .cargo/config.toml)");
    let libtorch = libtorch.trim_end_matches('/');
    let mut ld = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();

    if !ld.is_empty() {
        ld.push(':');
    }
    ld.push_str(&format!("{libtorch}/lib"));
    println!("cargo:rustc-env=LD_LIBRARY_PATH={ld}");
    // println!("cargo:rerun-if-env-changed=LIBTORCH");

    println!("cargo:rustc-env=CARGO_TARGET_DIR=target"); //https://stackoverflow.com/a/73543282/656449 - avoid rebuild all deps
    //TODO write to OUT_DIR + include!(concat!(env!("OUT_DIR"), "/version.rs"));
    fs::write("src/version.rs", format!("pub const VERSION:&'static str = \"{}\";", Utc::now())).expect("Unable to write file");
}
