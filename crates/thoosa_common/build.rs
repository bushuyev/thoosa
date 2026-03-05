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
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    println!("cargo:rustc-env=CARGO_TARGET_DIR=target"); //https://stackoverflow.com/a/73543282/656449 - avoid rebuild all deps

    let (debug_plots_mode, debug_plots_prefixes) = parse_debug_plots_rustflags(std::env::var("CARGO_ENCODED_RUSTFLAGS").ok().as_deref());
    println!("cargo:rustc-env=THO_DEBUG_PLOTS_MODE={debug_plots_mode}");
    println!("cargo:rustc-env=THO_DEBUG_PLOTS_PREFIXES={debug_plots_prefixes}");
}

fn parse_debug_plots_rustflags(encoded_rustflags: Option<&str>) -> (&'static str, String) {
    let Some(encoded_rustflags) = encoded_rustflags else {
        return ("off", String::new());
    };

    let flags = encoded_rustflags.split('\x1f').collect::<Vec<_>>();
    let mut has_plain_flag = false;
    let mut prefixes = Vec::new();
    let mut i = 0usize;
    while i < flags.len() {
        let token = flags[i];
        let cfg_spec = if token == "--cfg" {
            i += 1;
            flags.get(i).copied()
        } else {
            token.strip_prefix("--cfg=")
        };

        if let Some(cfg_spec) = cfg_spec {
            if cfg_spec == "tho_debug_plots" {
                has_plain_flag = true;
            } else if let Some(value) = cfg_spec.strip_prefix("tho_debug_plots=") {
                let value = strip_optional_quotes(value.trim());
                prefixes.extend(
                    value
                        .split(',')
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string()),
                );
            }
        }
        i += 1;
    }

    if has_plain_flag {
        ("all", String::new())
    } else if !prefixes.is_empty() {
        ("prefixes", prefixes.join(","))
    } else {
        ("off", String::new())
    }
}

fn strip_optional_quotes(s: &str) -> &str {
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::parse_debug_plots_rustflags;

    #[test]
    fn parse_debug_plots_plain_flag_enables_all() {
        let (mode, prefixes) = parse_debug_plots_rustflags(Some("--cfg\x1ftho_debug_plots"));
        assert_eq!(mode, "all");
        assert_eq!(prefixes, "");
    }

    #[test]
    fn parse_debug_plots_prefixes_value() {
        let (mode, prefixes) = parse_debug_plots_rustflags(Some("--cfg\x1ftho_debug_plots=batch_motif_,batch_random_motif_in_batch_"));
        assert_eq!(mode, "prefixes");
        assert_eq!(prefixes, "batch_motif_,batch_random_motif_in_batch_");
    }
}
