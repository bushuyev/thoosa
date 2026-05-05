use std::{
    fs,
    io,
    path::{Path, PathBuf},
};

#[derive(Debug)]
pub struct TmpFile {
    path: PathBuf,
}

impl TmpFile {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }



    /// Remove immediately and disarm the guard.
    pub fn remove_now(self) -> io::Result<()> {

        match fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }
}

impl Drop for TmpFile {
    fn drop(&mut self) {

        match fs::remove_file(&self.path) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => {
                // Avoid panicking in Drop.
                eprintln!("failed to remove file {:?}: {e}", self.path);
            }
        }
    }
}

impl AsRef<Path> for TmpFile {
    fn as_ref(&self) -> &Path {
        self.path.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        env,
        fs,
        io,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn removes_file_on_drop() -> io::Result<()> {
        let path = env::temp_dir().join(format!(
            "remove_file_on_drop_test_{}_{}.tmp",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        {
            fs::write(&path, b"temporary test file")?;
            assert!(path.exists());

            let _guard = TmpFile::new(&path);

            assert!(path.exists());
        } // `_guard` is dropped here

        assert!(!path.exists(), "file should have been removed: {path:?}");

        Ok(())
    }
}