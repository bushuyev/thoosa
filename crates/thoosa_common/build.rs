fn main() {

    let mut ld = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();

    if !ld.is_empty() { ld.push(':'); }
    ld.push_str("/home/bu/dvl/torch/libtorch-2.7.1/lib");
    println!("cargo:rustc-env=LD_LIBRARY_PATH={ld}");

    println!("cargo:rustc-env=CARGO_TARGET_DIR=target");//https://stackoverflow.com/a/73543282/656449 - avoid rebuild all deps
}
