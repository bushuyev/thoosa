
pub trait RunConfigSource {
    fn to_run_configs(&self) -> Vec<RunConfigs>;
}

pub struct RunConfigs {

}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it() {

    }
}
