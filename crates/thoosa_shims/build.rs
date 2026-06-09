use cmake::Config;
use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn main() {
    let libtorch = env::var("LIBTORCH")
        .expect("LIBTORCH must be set, e.g. via .cargo/config.toml");

    let libtorch = libtorch.trim_end_matches('/');
    let cuda_root = "/usr/local/cuda";

    let dst = Config::new("src-cpp")
        .define(
            "Torch_DIR",
            format!("{libtorch}/share/cmake/Torch"),
        )
        .define("CMAKE_CUDA_ARCHITECTURES", "89")
        .define("CUDAToolkit_ROOT", cuda_root)
        .define("CMAKE_CUDA_COMPILER", format!("{cuda_root}/bin/nvcc"))
        .define("CMAKE_BUILD_TYPE", "Debug")
        .define(
            "CUDA_CUDART_LIBRARY",
            format!("{cuda_root}/lib64/libcudart.so"),
        )
        .define(
            "CUDA_NVTX_LIBRARY",
            format!("{cuda_root}/lib64/libnvToolsExt.so"),
        )
        .define(
            "CUDA_driver_LIBRARY",
            "/usr/lib/x86_64-linux-gnu/libcuda.so",
        )
        .generator("Ninja")
        .build_target("thoosa_shims")
        .build();

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Example OUT_DIR:
    //
    // target/x86_64-unknown-linux-gnu/release-opt1/build/thoosa_shims-HASH/out
    //
    // ancestors:
    // 0: .../out
    // 1: .../thoosa_shims-HASH
    // 2: .../build
    // 3: target/x86_64-unknown-linux-gnu/release-opt1
    //
    // This works for custom profiles, unlike env::var("PROFILE").
    let cargo_profile_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("unexpected OUT_DIR layout")
        .to_path_buf();

    let cmake_build_dir = dst.join("build");

    copy_file(
        cmake_build_dir.join("libthoosa_shims.so"),
        cargo_profile_dir.join("libthoosa_shims.so"),
    );

    copy_file(
        cmake_build_dir.join("compile_commands.json"),
        cargo_profile_dir.join("compile_commands.json"),
    );

    println!(
        "cargo:rustc-link-search=native={}",
        cmake_build_dir.display()
    );

    println!("cargo:rustc-link-lib=dylib=thoosa_shims");

    println!("cargo:rustc-link-search=native={libtorch}/lib");
    println!("cargo:rustc-link-lib=dylib=torch_cuda");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=c10_cuda");

    // Let the final binary find libthoosa_shims.so in its own directory.
    //
    // This is important when running directly:
    //
    // target/x86_64-unknown-linux-gnu/release-opt1/your_binary
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");

    // Let the binary find libtorch libs.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{libtorch}/lib");

    println!("cargo:rerun-if-env-changed=LIBTORCH");

    rerun_if_changed("src-cpp/CMakeLists.txt");
    rerun_if_changed("src-cpp/stream.cpp");
    rerun_if_changed("src-cpp/memcpy.cpp");
    rerun_if_changed("src-cpp/context.cpp");
    rerun_if_changed("src-cpp/device_stats.cpp");
    rerun_if_changed("src-cpp/errors.cpp");
    rerun_if_changed("src-cpp/errors.h");
}

fn copy_file(src: impl AsRef<Path>, dst: impl AsRef<Path>) {
    let src = src.as_ref();
    let dst = dst.as_ref();

    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|e| {
            panic!(
                "failed to create destination directory {}: {e}",
                parent.display()
            )
        });
    }

    fs::copy(src, dst).unwrap_or_else(|e| {
        panic!(
            "failed to copy {} -> {}: {e}",
            src.display(),
            dst.display()
        )
    });
}

fn rerun_if_changed(path: &str) {
    println!("cargo:rerun-if-changed={path}");
}