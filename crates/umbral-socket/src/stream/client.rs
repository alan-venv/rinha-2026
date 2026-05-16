use std::io;
use std::os::unix::net::UnixStream;

use crate::stream::protocol::{
    MethodId, UmbralConfig, UmbralStatus, read_response_sync, write_request_sync,
};

pub struct UmbralClient {
    stream: UnixStream,
    config: UmbralConfig,
}

impl UmbralClient {
    pub fn new(socket: &str) -> io::Result<Self> {
        Self::with_config(socket, UmbralConfig::default())
    }

    pub fn with_config(socket: &str, config: UmbralConfig) -> io::Result<Self> {
        let stream = UnixStream::connect(socket)?;
        stream.set_read_timeout(Some(config.read_timeout))?;
        stream.set_write_timeout(Some(config.write_timeout))?;

        Ok(Self { stream, config })
    }

    pub fn send(&mut self, method: MethodId, payload: &[u8]) -> io::Result<Vec<u8>> {
        if payload.len() > self.config.max_payload_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "payload too large",
            ));
        }

        write_request_sync(&mut self.stream, method, payload)?;
        let (status, response) = read_response_sync(&mut self.stream, self.config.max_payload_len)?;
        if status == UmbralStatus::Ok {
            return Ok(response);
        }
        Err(request_error(status))
    }
}

fn request_error(status: UmbralStatus) -> io::Error {
    io::Error::other(format!("umbral request failed with status {status:?}"))
}
