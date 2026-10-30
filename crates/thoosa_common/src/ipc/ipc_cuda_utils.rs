//! Legacy per-buffer tensor staging helpers backed by anonymous Linux memfd objects.
//!
//! The active IPC copy protocol maps the parent allocator's arena once and sends
//! tensor offsets. These helpers remain for their focused low-level tests and any
//! callers that still need standalone shared mappings.

use std::ffi::c_void;
use std::io;
use std::mem::{size_of, zeroed};
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};

use tokio::net::UnixStream;







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
