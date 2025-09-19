fn main() {

    let mut ld = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();

    // panic!("ld={ld:?}");

    if !ld.is_empty() { ld.push(':'); }
    ld.push_str("/home/bu/dvl/torch/libtorch-2.7.1/lib");
    // ld.push_str("/home/bu/dvl/torch/src/pytorch/pytorch-2.7.1-debug-dsa-install/lib/");
    println!("cargo:rustc-env=LD_LIBRARY_PATH={ld}");
    println!("cargo:rustc-env=CARGO_TARGET_DIR=target");//https://stackoverflow.com/a/73543282/656449 - avoid rebuild all deps


    // let libtorch = "/home/bu/dvl/torch/libtorch-2.7.1/lib"; //std::env::var("LIBTORCH").expect("set LIBTORCH to LibTorch root");
    // // panic!("libtorch={libtorch}");
    // println!("cargo:rustc-link-search=native={}/lib", libtorch);
    // println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", libtorch);
    // println!("cargo:rustc-link-arg=-Wl,--enable-new-dtags");
}
