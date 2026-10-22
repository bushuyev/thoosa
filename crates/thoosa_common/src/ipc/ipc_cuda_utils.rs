//! Legacy per-buffer tensor staging helpers backed by anonymous Linux memfd objects.
//!
//! The active IPC copy protocol maps the parent allocator's arena once and sends
//! tensor offsets. These helpers remain for their focused low-level tests and any
//! callers that still need standalone shared mappings.

use std::ffi::c_void;
use std::fmt::Debug;
use std::io;
use std::mem::{size_of, zeroed};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};

use serde::{Deserialize, Serialize};
use tch::{Kind, Tensor};
use tokio::net::UnixStream;

use thoosa_common::ipc::tensor_info::IpcTensorKind;
use thoosa_shared::ThoError;


fn invalid(message: impl Into<String>) -> ThoError {
    ThoError::new(message)
}




/// Sends one duplicated reference to `fd` over a connected Unix domain socket.
pub async fn send_fd(stream: &UnixStream, fd: BorrowedFd<'_>) -> io::Result<()> {
    loop {
        stream.writable().await?;
        match send_fd_once(stream.as_raw_fd(), fd.as_raw_fd()) {
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => continue,
            result => return result,
        }
    }
}

/// Receives one fd sent with [`send_fd`] and returns independent ownership of it.
pub async fn recv_fd(stream: &UnixStream) -> io::Result<OwnedFd> {
    loop {
        stream.readable().await?;
        match recv_fd_once(stream.as_raw_fd()) {
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => continue,
            result => return result,
        }
    }
}

pub fn send_fd_once(socket_fd: RawFd, fd: RawFd) -> io::Result<()> {
    let marker = [0x46u8];
    let mut iov = libc::iovec { iov_base: marker.as_ptr().cast_mut().cast::<c_void>(), iov_len: marker.len() };
    let control_len = unsafe {
        // SAFETY: CMSG_SPACE is a pure size calculation.
        libc::CMSG_SPACE(size_of::<libc::c_int>() as u32) as usize
    };
    let mut control = vec![0usize; control_len.div_ceil(size_of::<usize>())];
    let mut message = unsafe {
        // SAFETY: zero is a valid initial representation for msghdr.
        zeroed::<libc::msghdr>()
    };
    message.msg_iov = &mut iov;
    message.msg_iovlen = 1;
    message.msg_control = control.as_mut_ptr().cast::<c_void>();
    message.msg_controllen = control_len;
    unsafe {
        // SAFETY: control is suitably aligned and sized by CMSG_SPACE. The cmsghdr
        // and its one descriptor payload are initialized before sendmsg reads them.
        let header = libc::CMSG_FIRSTHDR(&message);
        if header.is_null() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "failed to allocate SCM_RIGHTS header"));
        }
        (*header).cmsg_level = libc::SOL_SOCKET;
        (*header).cmsg_type = libc::SCM_RIGHTS;
        (*header).cmsg_len = libc::CMSG_LEN(size_of::<libc::c_int>() as u32) as usize;
        std::ptr::write(libc::CMSG_DATA(header).cast::<libc::c_int>(), fd);
        let sent = libc::sendmsg(socket_fd, &message, libc::MSG_NOSIGNAL);
        if sent < 0 {
            return Err(io::Error::last_os_error());
        }
        if sent != marker.len() as isize {
            return Err(io::Error::new(io::ErrorKind::WriteZero, format!("SCM_RIGHTS marker write was partial: {sent}")));
        }
    }
    Ok(())
}

fn recv_fd_once(socket_fd: RawFd) -> io::Result<OwnedFd> {
    let mut marker = [0u8; 1];
    let mut iov = libc::iovec { iov_base: marker.as_mut_ptr().cast::<c_void>(), iov_len: marker.len() };
    let control_len = unsafe {
        // SAFETY: CMSG_SPACE is a pure size calculation.
        libc::CMSG_SPACE(size_of::<libc::c_int>() as u32) as usize
    };
    let mut control = vec![0usize; control_len.div_ceil(size_of::<usize>())];
    let mut message = unsafe {
        // SAFETY: zero is a valid initial representation for msghdr.
        zeroed::<libc::msghdr>()
    };
    message.msg_iov = &mut iov;
    message.msg_iovlen = 1;
    message.msg_control = control.as_mut_ptr().cast::<c_void>();
    message.msg_controllen = control_len;
    let received = unsafe {
        // SAFETY: all message buffers are valid and writable for their declared sizes.
        libc::recvmsg(socket_fd, &mut message, libc::MSG_CMSG_CLOEXEC)
    };
    if received < 0 {
        return Err(io::Error::last_os_error());
    }
    if received == 0 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "socket closed while receiving memfd"));
    }
    if marker[0] != 0x46 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("invalid SCM_RIGHTS marker: {}", marker[0])));
    }
    if message.msg_flags & libc::MSG_CTRUNC != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "SCM_RIGHTS control message was truncated"));
    }
    let mut received_fd = None;
    unsafe {
        // SAFETY: recvmsg initialized the ancillary buffer; CMSG_* walks only within
        // msg_controllen. Each received descriptor is adopted exactly once.
        let mut header = libc::CMSG_FIRSTHDR(&message);
        while !header.is_null() {
            if (*header).cmsg_level == libc::SOL_SOCKET && (*header).cmsg_type == libc::SCM_RIGHTS {
                let raw_fd = std::ptr::read(libc::CMSG_DATA(header).cast::<libc::c_int>());
                let owned = OwnedFd::from_raw_fd(raw_fd);
                if received_fd.replace(owned).is_some() {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "received more than one fd"));
                }
            }
            header = libc::CMSG_NXTHDR(&message, header);
        }
    }
    received_fd.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "SCM_RIGHTS message contained no fd"))
}

#[cfg(test)]
mod tests {
    use anyhow::Context as _;
    use std::path::PathBuf;
    use std::process::Stdio;

    use tch::Device::Cpu;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixListener;
    use tokio::process::{Child, Command};

    use thoosa_shims::context::ThoCudaContext;

    use super::*;

    const TEST_SOCKET_ENV: &str = "THOOSA_MEMFD_COPY_TEST_SOCKET";
    const TEST_MODE_ENV: &str = "THOOSA_MEMFD_COPY_TEST_MODE";
    const TEST_FRAME_MAX: usize = 16 * 1024 * 1024;


    async fn write_frame(stream: &mut UnixStream, payload: &[u8]) -> Result<(), ThoError> {
        if payload.len() > u32::MAX as usize {
            return Err(invalid(format!("test frame too large: {}", payload.len())));
        }
        stream.write_all(&(payload.len() as u32).to_le_bytes()).await?;
        stream.write_all(payload).await?;
        stream.flush().await?;
        Ok(())
    }

    async fn read_frame(stream: &mut UnixStream) -> Result<Vec<u8>, ThoError> {
        let mut len = [0u8; 4];
        stream.read_exact(&mut len).await?;
        let len = u32::from_le_bytes(len) as usize;
        if len > TEST_FRAME_MAX {
            return Err(invalid(format!("test frame too large: {len}")));
        }
        let mut payload = vec![0u8; len];
        stream.read_exact(&mut payload).await?;
        Ok(payload)
    }

    async fn write_json<T: Serialize>(stream: &mut UnixStream, value: &T) -> Result<(), ThoError> {
        write_frame(stream, &serde_json::to_vec(value)?).await
    }

    async fn read_json<T: for<'de> Deserialize<'de>>(stream: &mut UnixStream) -> Result<T, ThoError> {
        Ok(serde_json::from_slice(&read_frame(stream).await?)?)
    }

    fn test_name() -> String {
        let module = module_path!();
        let module = module.split_once("::").map(|(_, module)| module).unwrap_or(module);
        format!("{module}::test_interprocess_h2d_d2h_and_worker_restart")
    }

    async fn spawn_worker(mode: &str) -> Result<(Child, UnixStream), ThoError> {
        let dir = tempfile::tempdir()?;
        let socket_path = dir.path().join("memfd-copy.sock");
        let listener = UnixListener::bind(&socket_path)?;
        let mut command = Command::new(std::env::current_exe()?);
        command
            .arg(test_name())
            .arg("--exact")
            .arg("--nocapture")
            .env(TEST_SOCKET_ENV, &socket_path)
            .env(TEST_MODE_ENV, mode)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        let mut child = command.spawn()?;
        let accepted = tokio::time::timeout(std::time::Duration::from_secs(10), listener.accept()).await;
        let stream = match accepted {
            Ok(Ok((stream, _))) => stream,
            Ok(Err(error)) => return Err(error.into()),
            Err(_) => {
                let status = child.try_wait()?;
                let _ = child.kill().await;
                return Err(invalid(format!("memfd copy worker failed to connect; status={status:?}")));
            }
        };
        drop(dir);
        Ok((child, stream))
    }

    async fn wait_worker(mut child: Child) -> Result<(), ThoError> {
        let status = tokio::time::timeout(std::time::Duration::from_secs(30), child.wait())
            .await
            .context("memfd copy worker did not exit")??;
        if !status.success() {
            return Err(invalid(format!("memfd copy worker exited with {status}")));
        }
        Ok(())
    }

    fn f32_bytes(values: &[f32]) -> &[u8] {
        unsafe {
            // SAFETY: f32 has no invalid bit patterns and the returned slice borrows
            // values for exactly its initialized byte extent.
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        }
    }

    fn tensor_bytes(tensor: &Tensor, byte_len: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; byte_len];
        unsafe {
            // SAFETY: tests call this only for contiguous CPU tensors whose validated
            // metadata requires exactly byte_len bytes.
            std::ptr::copy_nonoverlapping(tensor.data_ptr().cast::<u8>(), bytes.as_mut_ptr(), byte_len);
        }
        bytes
    }

}
