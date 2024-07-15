use std::fmt::Debug;
use std::fs;
use chrono::prelude::*;




fn main() {
    fs::write("src/version.rs", format!("pub const VERSION:&'static str = \"{}\";", Utc::now())).expect("Unable to write file");
}
