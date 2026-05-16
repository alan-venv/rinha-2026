use std::collections::HashMap;
use std::io::{self, IoSlice, Read, Result, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixListener as StdUnixListener;
use std::path::Path;
use std::slice;

use mio::net::{UnixListener, UnixStream};
use mio::{Events, Interest, Poll, Token};

use super::protocol::{MethodId, REQUEST_HEADER_LEN, UmbralConfig, UmbralStatus};

const SERVER: Token = Token(0);
const CONNECTION_START: usize = 1;

pub enum UmbralResponse {
    Empty,
    RequestPayload,
    ResponseBuffer,
    Static(&'static [u8]),
}

type Handler<S> = fn(&S, &[u8], &mut Vec<u8>) -> io::Result<UmbralResponse>;

pub struct UmbralServer<S> {
    state: S,
    handlers: [Option<Handler<S>>; 256],
    config: UmbralConfig,
}

struct Connection {
    stream: UnixStream,
    state: ConnectionState,
    close_after_write: bool,
    header: [u8; REQUEST_HEADER_LEN],
    header_offset: usize,
    request: Vec<u8>,
    request_len: usize,
    response: Vec<u8>,
    write_header: [u8; REQUEST_HEADER_LEN],
    write_header_offset: usize,
    write_payload: WritePayload,
    write_payload_offset: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConnectionState {
    ReadingHeader,
    ReadingBody,
    Writing,
}

#[derive(Clone, Copy)]
enum WritePayload {
    Empty,
    Request,
    Response,
    Static(&'static [u8]),
}

impl<S> UmbralServer<S> {
    pub fn new(state: S) -> Self {
        Self::with_config(state, UmbralConfig::default())
    }

    pub fn with_config(state: S, config: UmbralConfig) -> Self {
        Self {
            state,
            handlers: std::array::from_fn(|_| None),
            config,
        }
    }

    pub fn route(mut self, method: MethodId, handler: Handler<S>) -> Self {
        self.handlers[method as usize] = Some(handler);
        self
    }

    pub fn run(self, socket: &str) -> Result<()> {
        let path = Path::new(socket);
        if path.exists() {
            std::fs::remove_file(path)?;
        }

        let std_listener = StdUnixListener::bind(path)?;
        std_listener.set_nonblocking(true)?;
        let permissions = std::fs::Permissions::from_mode(self.config.socket_permissions);
        std::fs::set_permissions(path, permissions)?;

        let mut listener = UnixListener::from_std(std_listener);
        let mut poll = Poll::new()?;
        poll.registry()
            .register(&mut listener, SERVER, Interest::READABLE)?;

        let mut events = Events::with_capacity(1024);
        let mut connections = HashMap::new();
        let mut next_token = CONNECTION_START;

        println!("Umbral Server listening on \"{}\"", socket);

        loop {
            poll.poll(&mut events, None)?;

            for event in events.iter() {
                let token = event.token();
                if token == SERVER {
                    accept_connections(
                        poll.registry(),
                        &mut listener,
                        &mut connections,
                        &mut next_token,
                    )?;
                    continue;
                }

                if event.is_readable() {
                    self.read_connection(poll.registry(), token, &mut connections)?;
                }
                if event.is_writable() {
                    flush_connection(poll.registry(), token, &mut connections)?;
                }
            }
        }
    }

    fn read_connection(
        &self,
        registry: &mio::Registry,
        token: Token,
        connections: &mut HashMap<Token, Connection>,
    ) -> Result<()> {
        loop {
            let Some(connection) = connections.get_mut(&token) else {
                return Ok(());
            };

            if connection.state == ConnectionState::Writing {
                return update_interest(registry, token, connections);
            }

            match read_request(connection, self.config.max_payload_len) {
                Ok(Some(method)) => {
                    self.handle_request(connection, method);
                    flush_connection(registry, token, connections)?;
                }
                Ok(None) => return update_interest(registry, token, connections),
                Err(error) if error.kind() == io::ErrorKind::InvalidData => {
                    if let Some(connection) = connections.get_mut(&token) {
                        prepare_status(connection, UmbralStatus::PayloadTooLarge, true);
                    }
                    flush_connection(registry, token, connections)?;
                    return Ok(());
                }
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                    remove_connection(registry, token, connections)?;
                    return Ok(());
                }
                Err(error) => return Err(error),
            };
        }
    }

    fn handle_request(&self, connection: &mut Connection, method: MethodId) {
        connection.response.clear();

        let Some(handler) = self.handlers[method as usize] else {
            prepare_status(connection, UmbralStatus::MethodNotFound, false);
            return;
        };

        match handler(&self.state, &connection.request, &mut connection.response) {
            Ok(response) => {
                let payload = match response {
                    UmbralResponse::Empty => WritePayload::Empty,
                    UmbralResponse::RequestPayload => WritePayload::Request,
                    UmbralResponse::ResponseBuffer => WritePayload::Response,
                    UmbralResponse::Static(bytes) => WritePayload::Static(bytes),
                };

                if payload_len(connection, payload) > self.config.max_payload_len {
                    prepare_status(connection, UmbralStatus::HandlerError, false);
                    return;
                }

                prepare_response(connection, UmbralStatus::Ok, payload, false);
            }
            Err(_) => {
                prepare_status(connection, UmbralStatus::HandlerError, false);
            }
        }
    }
}

impl Connection {
    fn new(stream: UnixStream) -> Self {
        Self {
            stream,
            state: ConnectionState::ReadingHeader,
            close_after_write: false,
            header: [0; REQUEST_HEADER_LEN],
            header_offset: 0,
            request: Vec::new(),
            request_len: 0,
            response: Vec::new(),
            write_header: [0; REQUEST_HEADER_LEN],
            write_header_offset: 0,
            write_payload: WritePayload::Empty,
            write_payload_offset: 0,
        }
    }

    fn reset_read(&mut self) {
        self.state = ConnectionState::ReadingHeader;
        self.close_after_write = false;
        self.header = [0; REQUEST_HEADER_LEN];
        self.header_offset = 0;
        self.request.clear();
        self.request_len = 0;
        self.write_header_offset = 0;
        self.write_payload = WritePayload::Empty;
        self.write_payload_offset = 0;
    }
}

fn accept_connections(
    registry: &mio::Registry,
    listener: &mut UnixListener,
    connections: &mut HashMap<Token, Connection>,
    next_token: &mut usize,
) -> Result<()> {
    loop {
        let (mut stream, _) = match listener.accept() {
            Ok(accepted) => accepted,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(()),
            Err(error) if error.raw_os_error() == Some(24) => return Ok(()),
            Err(error) => return Err(error),
        };

        let token = Token(*next_token);
        *next_token += 1;
        registry.register(&mut stream, token, Interest::READABLE)?;
        connections.insert(token, Connection::new(stream));
    }
}

fn read_request(connection: &mut Connection, max_payload_len: usize) -> Result<Option<MethodId>> {
    loop {
        match connection.state {
            ConnectionState::ReadingHeader => {
                while connection.header_offset < REQUEST_HEADER_LEN {
                    match connection
                        .stream
                        .read(&mut connection.header[connection.header_offset..])
                    {
                        Ok(0) => {
                            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "eof"));
                        }
                        Ok(read) => connection.header_offset += read,
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            return Ok(None);
                        }
                        Err(error) => return Err(error),
                    }
                }

                connection.request_len = u32::from_be_bytes([
                    connection.header[1],
                    connection.header[2],
                    connection.header[3],
                    connection.header[4],
                ]) as usize;

                if connection.request_len > max_payload_len {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "payload too large",
                    ));
                }

                connection.request.clear();
                connection.request.reserve(connection.request_len);
                connection.state = ConnectionState::ReadingBody;

                if connection.request_len == 0 {
                    return Ok(Some(connection.header[0]));
                }
            }
            ConnectionState::ReadingBody => {
                while connection.request.len() < connection.request_len {
                    match read_into_spare(
                        &mut connection.stream,
                        &mut connection.request,
                        connection.request_len,
                    ) {
                        Ok(0) => {
                            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "eof"));
                        }
                        Ok(_) => {}
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            return Ok(None);
                        }
                        Err(error) => return Err(error),
                    }
                }

                return Ok(Some(connection.header[0]));
            }
            ConnectionState::Writing => return Ok(None),
        }
    }
}

fn read_into_spare(
    stream: &mut UnixStream,
    payload: &mut Vec<u8>,
    target_len: usize,
) -> Result<usize> {
    let remaining = target_len - payload.len();
    payload.reserve(remaining);

    let before = payload.len();
    let buffer = unsafe { slice::from_raw_parts_mut(payload.as_mut_ptr().add(before), remaining) };
    let read = stream.read(buffer)?;

    unsafe {
        payload.set_len(before + read);
    }

    Ok(read)
}

fn prepare_status(connection: &mut Connection, status: UmbralStatus, close: bool) {
    prepare_response(connection, status, WritePayload::Empty, close);
}

fn prepare_response(
    connection: &mut Connection,
    status: UmbralStatus,
    payload: WritePayload,
    close: bool,
) {
    let len = payload_len(connection, payload) as u32;
    connection.write_header[0] = status as u8;
    connection.write_header[1..].copy_from_slice(&len.to_be_bytes());
    connection.write_header_offset = 0;
    connection.write_payload = payload;
    connection.write_payload_offset = 0;
    connection.close_after_write = close;
    connection.state = ConnectionState::Writing;
}

fn flush_connection(
    registry: &mio::Registry,
    token: Token,
    connections: &mut HashMap<Token, Connection>,
) -> Result<()> {
    let Some(connection) = connections.get_mut(&token) else {
        return Ok(());
    };

    while connection.write_header_offset < connection.write_header.len()
        || connection.write_payload_offset < payload_len(connection, connection.write_payload)
    {
        let written = write_pending(connection)?;
        if written == 0 {
            break;
        }
    }

    if connection.write_header_offset == connection.write_header.len()
        && connection.write_payload_offset == payload_len(connection, connection.write_payload)
    {
        if connection.close_after_write {
            remove_connection(registry, token, connections)?;
            return Ok(());
        }

        if let Some(connection) = connections.get_mut(&token) {
            connection.reset_read();
        }
    }

    update_interest(registry, token, connections)
}

fn write_pending(connection: &mut Connection) -> Result<usize> {
    if connection.write_header_offset == connection.write_header.len() {
        return write_payload(connection);
    }

    let payload_len = payload_len(connection, connection.write_payload);
    if connection.write_payload_offset == payload_len {
        return match connection
            .stream
            .write(&connection.write_header[connection.write_header_offset..])
        {
            Ok(written) => {
                connection.write_header_offset += written;
                Ok(written)
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => Ok(0),
            Err(error) => Err(error),
        };
    }

    let written = {
        let header = &connection.write_header[connection.write_header_offset..];
        let payload = match connection.write_payload {
            WritePayload::Empty => b"".as_slice(),
            WritePayload::Request => connection.request.as_slice(),
            WritePayload::Response => connection.response.as_slice(),
            WritePayload::Static(bytes) => bytes,
        };
        let payload = &payload[connection.write_payload_offset..];
        let stream = &mut connection.stream;

        match stream.write_vectored(&[IoSlice::new(header), IoSlice::new(payload)]) {
            Ok(written) => written,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(0),
            Err(error) => return Err(error),
        }
    };

    if written == 0 {
        return Err(io::Error::new(
            io::ErrorKind::WriteZero,
            "failed to write umbral frame",
        ));
    }

    advance_write_offsets(connection, payload_len, written);
    Ok(written)
}

fn write_payload(connection: &mut Connection) -> Result<usize> {
    let written = {
        let payload = match connection.write_payload {
            WritePayload::Empty => b"".as_slice(),
            WritePayload::Request => connection.request.as_slice(),
            WritePayload::Response => connection.response.as_slice(),
            WritePayload::Static(bytes) => bytes,
        };
        let payload = &payload[connection.write_payload_offset..];
        let stream = &mut connection.stream;

        match stream.write(payload) {
            Ok(written) => written,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(0),
            Err(error) => return Err(error),
        }
    };

    connection.write_payload_offset += written;
    Ok(written)
}

fn advance_write_offsets(connection: &mut Connection, payload_len: usize, written: usize) {
    let remaining_header = connection.write_header.len() - connection.write_header_offset;
    if written < remaining_header {
        connection.write_header_offset += written;
    } else {
        connection.write_header_offset = connection.write_header.len();
        connection.write_payload_offset =
            (connection.write_payload_offset + written - remaining_header).min(payload_len);
    }
}

fn update_interest(
    registry: &mio::Registry,
    token: Token,
    connections: &mut HashMap<Token, Connection>,
) -> Result<()> {
    let Some(connection) = connections.get_mut(&token) else {
        return Ok(());
    };
    let interest = if connection.state == ConnectionState::Writing {
        Interest::WRITABLE
    } else {
        Interest::READABLE
    };
    registry.reregister(&mut connection.stream, token, interest)
}

fn remove_connection(
    registry: &mio::Registry,
    token: Token,
    connections: &mut HashMap<Token, Connection>,
) -> Result<()> {
    if let Some(mut connection) = connections.remove(&token) {
        registry.deregister(&mut connection.stream)?;
    }
    Ok(())
}

fn payload_len(connection: &Connection, payload: WritePayload) -> usize {
    payload_slice(connection, payload).len()
}

fn payload_slice(connection: &Connection, payload: WritePayload) -> &[u8] {
    match payload {
        WritePayload::Empty => b"",
        WritePayload::Request => connection.request.as_slice(),
        WritePayload::Response => connection.response.as_slice(),
        WritePayload::Static(bytes) => bytes,
    }
}
