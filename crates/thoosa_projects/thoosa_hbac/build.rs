#[path = "../../build_support.rs"]
mod build_support;

fn main() {
    println!("cargo:rerun-if-changed=../../build_support.rs");
    build_support::configure_libtorch();
}
