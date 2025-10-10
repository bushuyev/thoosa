use std::{env, fs};
use cmake::Config;
use std::path::PathBuf;

fn main() {
    let libtorch = "/home/bu/dvl/torch/libtorch-2.7.1";
    let cuda_root = "/usr/local/cuda";

    let dst = Config::new("src-cpp")
        .define(
            "Torch_DIR",
            format!("{}/share/cmake/Torch", libtorch),
        )
        .define("CMAKE_CUDA_ARCHITECTURES", "89")
        .define("CUDAToolkit_ROOT", "/usr/local/cuda")
        .define("CMAKE_CUDA_COMPILER", "/usr/local/cuda/bin/nvcc")
        .define("CMAKE_BUILD_TYPE", "Debug")
        .define("CUDA_CUDART_LIBRARY", format!("{cuda_root}/lib64/libcudart.so"))
        .define("CUDA_NVTX_LIBRARY", format!("{cuda_root}/lib64/libnvToolsExt.so"))   // or libnvtx3.so
        .define("CUDA_driver_LIBRARY", "/usr/lib/x86_64-linux-gnu/libcuda.so")
        .generator("Ninja")
        .build_target("thoosa_shims")
        .build();

    let out_dir  = PathBuf::from(env::var("OUT_DIR").unwrap());

    let profile_dir = PathBuf::from(&env::var("OUT_DIR").unwrap())
        .ancestors().nth(5).unwrap()
        .join(env::var("PROFILE").unwrap())
        .to_path_buf();

    let deps = deps_dir();
    fs::create_dir_all(&deps).unwrap();
    fs::copy(&out_dir.join("build/libthoosa_shims.so"), deps.join(profile_dir.join("libthoosa_shims.so"))).unwrap();
    fs::copy(&out_dir.join("build/compile_commands.json"), deps.join(profile_dir.join("compile_commands.json"))).unwrap();


    let lib_dir: PathBuf = dst.join("build");

    println!("cargo:rustc-link-arg=-Wl,-rpath,{libtorch}/lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=thoosa_shims");
    println!("cargo:rustc-link-search=native={libtorch}/lib");
    println!("cargo:rustc-link-lib=dylib=torch_cuda");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=c10_cuda");

    println!("cargo:rerun-if-changed=src-cpp/stream.cpp");
    println!("cargo:rerun-if-changed=src-cpp/memcpy.cpp");

}


fn deps_dir() -> PathBuf {
    // where Cargo puts *all* build artefacts
    let target_root = env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target"));

    let triple  = env::var("TARGET").unwrap();   // always set
    let profile = env::var("PROFILE").unwrap();  // debug / release …

    // target/<triple>/<profile>/deps
    target_root.join(triple).join(profile)//.join("deps")
}
