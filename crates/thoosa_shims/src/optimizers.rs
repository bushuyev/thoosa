

unsafe extern "C" {
    fn tho_snapshot_adam_states();
}

#[cfg(test)]
mod tests {
    use crate::optimizers::tho_snapshot_adam_states;

    #[test]
    fn test_tho_snapshot_adam_states(){
        unsafe {
            tho_snapshot_adam_states();
        }
    }

}