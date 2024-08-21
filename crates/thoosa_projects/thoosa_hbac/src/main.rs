use std::time::Instant;
use log::{error, info};
use thoosa_main::run_cli;

fn main()  {
    let t1 = Instant::now();

    match run_cli() {

        Ok(_) => {
            info!("All done in {} mls", t1.elapsed().as_millis());
        }

        Err(e) => {
            error!("failed with {:?} in {} mls", e, t1.elapsed().as_millis());
        }
    }
    std::process::exit(0x0000);//for jupyter
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_0(){

    }
}