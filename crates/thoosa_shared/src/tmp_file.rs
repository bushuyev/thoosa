use log::debug;
use serde::{Deserialize, Serialize};
use std::{
    fs, io,
    path::{Path, PathBuf},
};

#[derive(Debug, Deserialize, Serialize)]
pub struct TmpFile {
    path: PathBuf,
    #[serde(skip, default = "armed")]
    armed: bool,
}

// Deserialized guards own cleanup just like freshly created guards.
fn armed() -> bool {
    true
}

impl TmpFile {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into(), armed: true }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Remove immediately and disarm the guard.
    pub fn remove_now(mut self) -> io::Result<()> {
        self.remove_once()
    }

    fn remove_once(&mut self) -> io::Result<()> {
        // Disarm before attempting I/O: even an error must not trigger a second
        // deletion of a path that another owner could have recreated.
        self.armed = false;

        match fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }
}

impl Drop for TmpFile {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        debug!("TmpFile::drop: self.path={:?}", self.path);

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
        env, fs, io,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn test_explicit_removal_does_not_delete_a_replacement() -> io::Result<()> {
        let path = env::temp_dir().join(format!("tmp-file-{}", uuid::Uuid::new_v4()));
        let mut guard = TmpFile::new(&path);
        fs::write(&path, b"original")?;
        guard.remove_once()?;
        fs::write(&path, b"replacement")?;
        drop(guard);
        assert_eq!(fs::read(&path)?, b"replacement");
        fs::remove_file(path)
    }

    #[test]
    fn removes_file_on_drop() -> io::Result<()> {
        let path = env::temp_dir().join(format!("remove_file_on_drop_test_{}_{}.tmp", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));

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
