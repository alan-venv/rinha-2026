use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::os::unix::net::UnixStream as StdUnixStream;
use std::time::Duration;

use mio::net::{TcpListener, TcpStream, UnixStream};
use mio::{Events, Interest, Poll, Token};
use rinha::controller;
use umbral_socket::stream::{RESPONSE_HEADER_LEN, UmbralStatus};

const DEFAULT_BIND: &str = "0.0.0.0:80";
const DEFAULT_UPSTREAMS: &str = "/sockets/service-1.sock,/sockets/service-2.sock";
const DEFAULT_CONNECTIONS_PER_UPSTREAM: usize = 4;
const READ_BUFFER: usize = 16 * 1024;
const MAX_HEADER_LEN: usize = 8 * 1024;
const MAX_BODY_LEN: usize = 64 * 1024;
const MAX_RESPONSE_BODY_LEN: usize = 64;
const SERVER: Token = Token(0);
const UPSTREAM_START: usize = 1;

const READY_RESPONSE: &[u8] =
    b"HTTP/1.1 200 OK\r\nContent-Length: 16\r\nConnection: keep-alive\r\n\r\nI was born ready";
const BAD_REQUEST: &[u8] =
    b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
const NOT_FOUND: &[u8] =
    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";
const SERVICE_UNAVAILABLE: &[u8] =
    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
const RESPONSE_PREFIX: &[u8] =
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ";
const RESPONSE_SUFFIX: &[u8] = b"\r\nConnection: keep-alive\r\n\r\n";

type Connections = Vec<Option<Connection>>;

struct Connection {
    stream: TcpStream,
    read_buffer: Vec<u8>,
    write_buffer: Vec<u8>,
    write_offset: usize,
    pending: bool,
    closing: bool,
}

struct UpstreamLane {
    stream: UnixStream,
    socket: String,
    token: Token,
    state: LaneState,
    client: Option<Token>,
    write_buffer: Vec<u8>,
    write_offset: usize,
    header: [u8; RESPONSE_HEADER_LEN],
    header_offset: usize,
    response: [u8; MAX_RESPONSE_BODY_LEN],
    response_offset: usize,
    response_len: usize,
    status: UmbralStatus,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LaneState {
    Idle,
    Writing,
    ReadingHeader,
    ReadingBody,
}

struct Job {
    client: Token,
    body_start: usize,
    body_len: usize,
    total_len: usize,
}

enum ParsedRequest {
    Ready,
    FraudScore {
        body_start: usize,
        body_len: usize,
        total_len: usize,
    },
    NotFound,
    BadRequest,
    Incomplete,
}

struct RequestHead {
    is_ready: bool,
    is_fraud_score: bool,
    content_length: usize,
}

fn main() -> io::Result<()> {
    let bind = std::env::var("LB_BIND").unwrap_or_else(|_| DEFAULT_BIND.to_string());
    let address = bind
        .parse::<SocketAddr>()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;

    let mut poll = Poll::new()?;
    let mut listener = TcpListener::bind(address)?;
    poll.registry()
        .register(&mut listener, SERVER, Interest::READABLE)?;

    let mut lanes = connect_lanes(
        poll.registry(),
        upstream_sockets(),
        connections_per_upstream(),
    )?;
    let mut idle_lanes = (0..lanes.len()).rev().collect::<Vec<_>>();
    let client_start = UPSTREAM_START + lanes.len();
    let mut pending_jobs = VecDeque::new();
    let mut events = Events::with_capacity(1024);
    let mut connections = Vec::new();
    let mut next_token = client_start;
    println!("LB: {bind}");

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
            } else if token.0 >= UPSTREAM_START && token.0 < client_start {
                let lane_index = token.0 - UPSTREAM_START;
                handle_lane_event(
                    poll.registry(),
                    lane_index,
                    &mut lanes,
                    &mut idle_lanes,
                    &mut pending_jobs,
                    &mut connections,
                    event.is_readable(),
                    event.is_writable(),
                )?;
            } else {
                if event.is_readable() {
                    read_connection(
                        poll.registry(),
                        token,
                        &mut connections,
                        &mut lanes,
                        &mut idle_lanes,
                        &mut pending_jobs,
                    )?;
                }
                if event.is_writable() {
                    flush_connection(poll.registry(), token, &mut connections)?;
                }
            }
        }
    }
}

fn connect_lanes(
    registry: &mio::Registry,
    sockets: Vec<String>,
    connections_per_upstream: usize,
) -> io::Result<Vec<UpstreamLane>> {
    let mut lanes = Vec::with_capacity(sockets.len() * connections_per_upstream);

    for socket in sockets {
        for _ in 0..connections_per_upstream {
            let token = Token(UPSTREAM_START + lanes.len());
            let mut stream = connect_upstream(&socket)?;
            registry.register(&mut stream, token, Interest::READABLE)?;
            lanes.push(UpstreamLane {
                stream,
                socket: socket.clone(),
                token,
                state: LaneState::Idle,
                client: None,
                write_buffer: Vec::with_capacity(1024),
                write_offset: 0,
                header: [0; RESPONSE_HEADER_LEN],
                header_offset: 0,
                response: [0; MAX_RESPONSE_BODY_LEN],
                response_offset: 0,
                response_len: 0,
                status: UmbralStatus::Ok,
            });
        }
    }

    Ok(lanes)
}

fn accept_connections(
    registry: &mio::Registry,
    listener: &mut TcpListener,
    connections: &mut Connections,
    next_token: &mut usize,
) -> io::Result<()> {
    loop {
        let (mut stream, _) = match listener.accept() {
            Ok(accepted) => accepted,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(()),
            Err(error) if error.raw_os_error() == Some(24) => return Ok(()),
            Err(error) => return Err(error),
        };

        stream.set_nodelay(true)?;
        let token = Token(*next_token);
        *next_token += 1;
        registry.register(&mut stream, token, Interest::READABLE)?;
        insert_connection(
            connections,
            token,
            Connection {
                stream,
                read_buffer: Vec::with_capacity(READ_BUFFER),
                write_buffer: Vec::with_capacity(128),
                write_offset: 0,
                pending: false,
                closing: false,
            },
        );
    }
}

fn read_connection(
    registry: &mio::Registry,
    token: Token,
    connections: &mut Connections,
    lanes: &mut [UpstreamLane],
    idle_lanes: &mut Vec<usize>,
    pending_jobs: &mut VecDeque<Job>,
) -> io::Result<()> {
    let Some(connection) = connection_mut(connections, token) else {
        return Ok(());
    };

    let mut buffer = [0u8; READ_BUFFER];
    loop {
        match connection.stream.read(&mut buffer) {
            Ok(0) => {
                remove_connection(registry, token, connections)?;
                return Ok(());
            }
            Ok(read) => connection.read_buffer.extend_from_slice(&buffer[..read]),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
            Err(error) => return Err(error),
        }
    }

    process_requests(
        registry,
        token,
        connections,
        lanes,
        idle_lanes,
        pending_jobs,
    )
}

fn process_requests(
    registry: &mio::Registry,
    token: Token,
    connections: &mut Connections,
    lanes: &mut [UpstreamLane],
    idle_lanes: &mut Vec<usize>,
    pending_jobs: &mut VecDeque<Job>,
) -> io::Result<()> {
    loop {
        let parsed = {
            let Some(connection) = connection_mut(connections, token) else {
                return Ok(());
            };
            if connection.pending || !connection.write_buffer.is_empty() {
                return update_interest(registry, token, connections);
            }
            parse_request(&mut connection.read_buffer)?
        };

        match parsed {
            ParsedRequest::Ready => {
                queue_response(registry, token, connections, READY_RESPONSE, false)?
            }
            ParsedRequest::NotFound => {
                queue_response(registry, token, connections, NOT_FOUND, false)?
            }
            ParsedRequest::BadRequest => {
                queue_response(registry, token, connections, BAD_REQUEST, true)?
            }
            ParsedRequest::Incomplete => return update_interest(registry, token, connections),
            ParsedRequest::FraudScore {
                body_start,
                body_len,
                total_len,
            } => {
                if let Some(connection) = connection_mut(connections, token) {
                    connection.pending = true;
                }
                assign_job(
                    registry,
                    lanes,
                    idle_lanes,
                    pending_jobs,
                    connections,
                    Job {
                        client: token,
                        body_start,
                        body_len,
                        total_len,
                    },
                )?;
                return update_interest(registry, token, connections);
            }
        }
    }
}

fn assign_job(
    registry: &mio::Registry,
    lanes: &mut [UpstreamLane],
    idle_lanes: &mut Vec<usize>,
    pending_jobs: &mut VecDeque<Job>,
    connections: &mut Connections,
    job: Job,
) -> io::Result<()> {
    let Some(index) = idle_lanes.pop() else {
        pending_jobs.push_back(job);
        return Ok(());
    };

    if !start_lane_job(registry, &mut lanes[index], connections, job)? {
        idle_lanes.push(index);
    }

    Ok(())
}

fn start_lane_job(
    registry: &mio::Registry,
    lane: &mut UpstreamLane,
    connections: &mut Connections,
    job: Job,
) -> io::Result<bool> {
    let Some(connection) = connection_mut(connections, job.client) else {
        return Ok(false);
    };
    let body_end = job.body_start + job.body_len;
    if connection.read_buffer.len() < job.total_len || body_end > job.total_len {
        return Ok(false);
    }

    lane.client = Some(job.client);
    lane.write_buffer.clear();
    lane.write_buffer.push(controller::FRAUD_SCORE_METHOD);
    lane.write_buffer
        .extend_from_slice(&(job.body_len as u32).to_be_bytes());
    lane.write_buffer
        .extend_from_slice(&connection.read_buffer[job.body_start..body_end]);
    drain_request(&mut connection.read_buffer, job.total_len);
    lane.write_offset = 0;
    lane.header = [0; RESPONSE_HEADER_LEN];
    lane.header_offset = 0;
    lane.response_offset = 0;
    lane.response_len = 0;
    lane.status = UmbralStatus::Ok;
    lane.state = LaneState::Writing;
    registry.reregister(&mut lane.stream, lane.token, Interest::WRITABLE)?;
    flush_lane_write(registry, lane)?;
    Ok(true)
}

fn handle_lane_event(
    registry: &mio::Registry,
    index: usize,
    lanes: &mut [UpstreamLane],
    idle_lanes: &mut Vec<usize>,
    pending_jobs: &mut VecDeque<Job>,
    connections: &mut Connections,
    readable: bool,
    writable: bool,
) -> io::Result<()> {
    if writable && lanes[index].state == LaneState::Writing {
        flush_lane_write(registry, &mut lanes[index])?;
    }

    if readable
        && matches!(
            lanes[index].state,
            LaneState::ReadingHeader | LaneState::ReadingBody
        )
    {
        read_lane_response(
            registry,
            index,
            lanes,
            idle_lanes,
            pending_jobs,
            connections,
        )?;
    }

    Ok(())
}

fn flush_lane_write(registry: &mio::Registry, lane: &mut UpstreamLane) -> io::Result<()> {
    while lane.write_offset < lane.write_buffer.len() {
        match lane.stream.write(&lane.write_buffer[lane.write_offset..]) {
            Ok(0) => break,
            Ok(written) => lane.write_offset += written,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(()),
            Err(error) => return Err(error),
        }
    }

    if lane.write_offset == lane.write_buffer.len() {
        lane.state = LaneState::ReadingHeader;
        registry.reregister(&mut lane.stream, lane.token, Interest::READABLE)?;
    }

    Ok(())
}

fn read_lane_response(
    registry: &mio::Registry,
    index: usize,
    lanes: &mut [UpstreamLane],
    idle_lanes: &mut Vec<usize>,
    pending_jobs: &mut VecDeque<Job>,
    connections: &mut Connections,
) -> io::Result<()> {
    let complete = {
        let lane = &mut lanes[index];
        loop {
            match lane.state {
                LaneState::ReadingHeader => {
                    match lane.stream.read(&mut lane.header[lane.header_offset..]) {
                        Ok(0) => {
                            return reconnect_lane(registry, index, lanes, idle_lanes, connections);
                        }
                        Ok(read) => {
                            lane.header_offset += read;
                            if lane.header_offset == RESPONSE_HEADER_LEN {
                                lane.status = UmbralStatus::try_from(lane.header[0])?;
                                lane.response_len = u32::from_be_bytes([
                                    lane.header[1],
                                    lane.header[2],
                                    lane.header[3],
                                    lane.header[4],
                                ]) as usize;
                                if lane.response_len > MAX_RESPONSE_BODY_LEN {
                                    return reconnect_lane(
                                        registry,
                                        index,
                                        lanes,
                                        idle_lanes,
                                        connections,
                                    );
                                }
                                lane.response_offset = 0;
                                lane.state = LaneState::ReadingBody;
                                if lane.response_len == 0 {
                                    break true;
                                }
                            }
                        }
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => break false,
                        Err(_) => {
                            return reconnect_lane(registry, index, lanes, idle_lanes, connections);
                        }
                    }
                }
                LaneState::ReadingBody => {
                    match lane
                        .stream
                        .read(&mut lane.response[lane.response_offset..lane.response_len])
                    {
                        Ok(0) => {
                            return reconnect_lane(registry, index, lanes, idle_lanes, connections);
                        }
                        Ok(read) => {
                            lane.response_offset += read;
                            if lane.response_offset == lane.response_len {
                                break true;
                            }
                        }
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => break false,
                        Err(_) => {
                            return reconnect_lane(registry, index, lanes, idle_lanes, connections);
                        }
                    }
                }
                _ => break false,
            }
        }
    };

    if complete {
        complete_lane(
            registry,
            index,
            lanes,
            idle_lanes,
            pending_jobs,
            connections,
        )?;
    }

    Ok(())
}

fn complete_lane(
    registry: &mio::Registry,
    index: usize,
    lanes: &mut [UpstreamLane],
    idle_lanes: &mut Vec<usize>,
    pending_jobs: &mut VecDeque<Job>,
    connections: &mut Connections,
) -> io::Result<()> {
    let (client, status) = {
        let lane = &mut lanes[index];
        let client = lane.client.take();
        let status = lane.status;
        (client, status)
    };

    if let Some(client) = client {
        if status == UmbralStatus::Ok {
            complete_client_json(
                registry,
                client,
                connections,
                &lanes[index].response[..lanes[index].response_len],
                false,
            )?;
        } else {
            complete_client_bytes(registry, client, connections, SERVICE_UNAVAILABLE, true)?;
        }
    }

    lanes[index].state = LaneState::Idle;
    registry.reregister(
        &mut lanes[index].stream,
        lanes[index].token,
        Interest::READABLE,
    )?;

    while let Some(job) = pending_jobs.pop_front() {
        if start_lane_job(registry, &mut lanes[index], connections, job)? {
            return Ok(());
        }
    }

    idle_lanes.push(index);
    Ok(())
}

fn reconnect_lane(
    registry: &mio::Registry,
    index: usize,
    lanes: &mut [UpstreamLane],
    idle_lanes: &mut Vec<usize>,
    connections: &mut Connections,
) -> io::Result<()> {
    let client = lanes[index].client.take();
    if let Some(client) = client {
        complete_client_bytes(registry, client, connections, SERVICE_UNAVAILABLE, true)?;
    }

    registry.deregister(&mut lanes[index].stream)?;
    lanes[index].stream = connect_upstream(&lanes[index].socket)?;
    lanes[index].state = LaneState::Idle;
    lanes[index].write_buffer.clear();
    lanes[index].write_offset = 0;
    lanes[index].response_len = 0;
    lanes[index].response_offset = 0;
    registry.register(
        &mut lanes[index].stream,
        lanes[index].token,
        Interest::READABLE,
    )?;
    idle_lanes.push(index);
    Ok(())
}

fn complete_client_json(
    registry: &mio::Registry,
    token: Token,
    connections: &mut Connections,
    payload: &[u8],
    close: bool,
) -> io::Result<()> {
    let Some(connection) = connection_mut(connections, token) else {
        return Ok(());
    };
    connection.pending = false;
    write_json_response_bytes(payload, &mut connection.write_buffer);
    connection.write_offset = 0;
    connection.closing = close;
    flush_connection(registry, token, connections)
}

fn complete_client_bytes(
    registry: &mio::Registry,
    token: Token,
    connections: &mut Connections,
    response: &[u8],
    close: bool,
) -> io::Result<()> {
    let Some(connection) = connection_mut(connections, token) else {
        return Ok(());
    };
    connection.pending = false;
    connection.write_buffer.clear();
    connection.write_buffer.extend_from_slice(response);
    connection.write_offset = 0;
    connection.closing = close;
    flush_connection(registry, token, connections)
}

fn queue_response(
    registry: &mio::Registry,
    token: Token,
    connections: &mut Connections,
    response: &[u8],
    close: bool,
) -> io::Result<()> {
    let Some(connection) = connection_mut(connections, token) else {
        return Ok(());
    };
    connection.write_buffer.clear();
    connection.write_buffer.extend_from_slice(response);
    connection.write_offset = 0;
    connection.closing = close;
    flush_connection(registry, token, connections)
}

fn flush_connection(
    registry: &mio::Registry,
    token: Token,
    connections: &mut Connections,
) -> io::Result<()> {
    let Some(connection) = connection_mut(connections, token) else {
        return Ok(());
    };

    while connection.write_offset < connection.write_buffer.len() {
        match connection
            .stream
            .write(&connection.write_buffer[connection.write_offset..])
        {
            Ok(0) => break,
            Ok(written) => connection.write_offset += written,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                return update_interest(registry, token, connections);
            }
            Err(error) => return Err(error),
        }
    }

    if connection.write_offset == connection.write_buffer.len() {
        connection.write_buffer.clear();
        connection.write_offset = 0;
        if connection.closing {
            remove_connection(registry, token, connections)?;
            return Ok(());
        }
    }

    update_interest(registry, token, connections)
}

fn update_interest(
    registry: &mio::Registry,
    token: Token,
    connections: &mut Connections,
) -> io::Result<()> {
    let Some(connection) = connection_mut(connections, token) else {
        return Ok(());
    };
    let interest = if connection.write_buffer.is_empty() {
        Interest::READABLE
    } else {
        Interest::READABLE.add(Interest::WRITABLE)
    };
    registry.reregister(&mut connection.stream, token, interest)
}

fn remove_connection(
    registry: &mio::Registry,
    token: Token,
    connections: &mut Connections,
) -> io::Result<()> {
    if let Some(mut connection) = take_connection(connections, token) {
        registry.deregister(&mut connection.stream)?;
    }
    Ok(())
}

fn insert_connection(connections: &mut Connections, token: Token, connection: Connection) {
    if connections.len() <= token.0 {
        connections.resize_with(token.0 + 1, || None);
    }
    connections[token.0] = Some(connection);
}

fn connection_mut(connections: &mut Connections, token: Token) -> Option<&mut Connection> {
    connections.get_mut(token.0)?.as_mut()
}

fn take_connection(connections: &mut Connections, token: Token) -> Option<Connection> {
    connections.get_mut(token.0)?.take()
}

fn parse_request(buffer: &mut Vec<u8>) -> io::Result<ParsedRequest> {
    let Some(header_end) = find_header_end(buffer) else {
        if buffer.len() >= MAX_HEADER_LEN {
            buffer.clear();
            return Ok(ParsedRequest::BadRequest);
        }
        return Ok(ParsedRequest::Incomplete);
    };

    let header = &buffer[..header_end];
    let request = parse_request_head(header)?;
    let total_len = header_end + 4 + request.content_length;

    if request.content_length > MAX_BODY_LEN {
        buffer.clear();
        return Ok(ParsedRequest::BadRequest);
    }

    if buffer.len() < total_len {
        return Ok(ParsedRequest::Incomplete);
    }

    if request.is_ready {
        drain_request(buffer, total_len);
        Ok(ParsedRequest::Ready)
    } else if request.is_fraud_score {
        Ok(ParsedRequest::FraudScore {
            body_start: header_end + 4,
            body_len: request.content_length,
            total_len,
        })
    } else {
        drain_request(buffer, total_len);
        Ok(ParsedRequest::NotFound)
    }
}

fn parse_request_head(header: &[u8]) -> io::Result<RequestHead> {
    let line_end = header
        .windows(2)
        .position(|window| window == b"\r\n")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid request line"))?;
    let request_line = &header[..line_end];
    let is_ready = request_line.starts_with(b"GET /ready ");
    let is_fraud_score = request_line.starts_with(b"POST /fraud-score ");

    let mut content_length = 0;
    for line in header[line_end + 2..].split(|byte| *byte == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if let Some(value) = header_value(line, b"Content-Length:") {
            content_length = parse_usize(value)?;
            break;
        }
        if let Some(value) = header_value(line, b"content-length:") {
            content_length = parse_usize(value)?;
            break;
        }
    }

    Ok(RequestHead {
        is_ready,
        is_fraud_score,
        content_length,
    })
}

fn connect_upstream(socket: &str) -> io::Result<UnixStream> {
    loop {
        match StdUnixStream::connect(socket) {
            Ok(stream) => {
                stream.set_nonblocking(true)?;
                return Ok(UnixStream::from_std(stream));
            }
            Err(_) => std::thread::sleep(Duration::from_millis(5)),
        }
    }
}

fn header_value<'a>(line: &'a [u8], name: &[u8]) -> Option<&'a [u8]> {
    line.get(..name.len())
        .filter(|candidate| *candidate == name)
        .map(|_| trim_spaces(&line[name.len()..]))
}

fn trim_spaces(mut value: &[u8]) -> &[u8] {
    while matches!(value.first(), Some(b' ' | b'\t')) {
        value = &value[1..];
    }
    value
}

fn parse_usize(value: &[u8]) -> io::Result<usize> {
    let mut output = 0_usize;
    for byte in value {
        if !byte.is_ascii_digit() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid content length",
            ));
        }
        output = output
            .checked_mul(10)
            .and_then(|value| value.checked_add((byte - b'0') as usize))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "content length overflow"))?;
    }
    Ok(output)
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn drain_request(buffer: &mut Vec<u8>, end: usize) {
    if end == buffer.len() {
        buffer.clear();
    } else {
        buffer.drain(..end);
    }
}

fn write_json_response_bytes(body: &[u8], output: &mut Vec<u8>) {
    output.clear();
    output.extend_from_slice(RESPONSE_PREFIX);
    push_usize(body.len(), output);
    output.extend_from_slice(RESPONSE_SUFFIX);
    output.extend_from_slice(body);
}

fn push_usize(mut value: usize, output: &mut Vec<u8>) {
    let mut digits = [0u8; 20];
    let mut index = digits.len();

    loop {
        index -= 1;
        digits[index] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }

    output.extend_from_slice(&digits[index..]);
}

fn upstream_sockets() -> Vec<String> {
    let sockets: Vec<String> = std::env::var("UPSTREAMS")
        .unwrap_or_else(|_| DEFAULT_UPSTREAMS.to_string())
        .split(',')
        .filter(|socket| !socket.is_empty())
        .map(str::to_string)
        .collect();

    if sockets.is_empty() {
        DEFAULT_UPSTREAMS.split(',').map(str::to_string).collect()
    } else {
        sockets
    }
}

fn connections_per_upstream() -> usize {
    std::env::var("UPSTREAM_CONNECTIONS_PER_WORKER")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_CONNECTIONS_PER_UPSTREAM)
}
