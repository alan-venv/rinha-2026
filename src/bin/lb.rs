use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::os::unix::net::UnixStream as StdUnixStream;
use std::time::Duration;

use mio::net::{TcpListener, TcpStream, UnixStream};
use mio::{Events, Interest, Poll, Token};
use rinha::controller;
use umbral_socket::stream::{RESPONSE_HEADER_LEN, UmbralStatus};

const BIND: &str = "0.0.0.0:80";
const UPSTREAM_SOCKETS: [&str; 2] = ["/sockets/service-1.sock", "/sockets/service-2.sock"];
const CONNECTIONS_PER_UPSTREAM: usize = 6;
const READ_BUFFER: usize = 16 * 1024;
const MAX_HEADER_LEN: usize = 1024;
const MAX_BODY_LEN: usize = 2048;
const MAX_RESPONSE_BODY_LEN: usize = 64;
const SERVER: Token = Token(0);
const UPSTREAM_START: usize = 1;

const READY_RESPONSE: &[u8] =
    b"HTTP/1.1 200 OK\r\nContent-Length: 16\r\nConnection: keep-alive\r\n\r\nI was born ready";
const ERROR_RESPONSE: &[u8] =
    b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
const BODY_SCORE_0: &[u8] = br#"{"approved":true,"fraud_score":0.0}"#;
const BODY_SCORE_02: &[u8] = br#"{"approved":true,"fraud_score":0.2}"#;
const BODY_SCORE_04: &[u8] = br#"{"approved":true,"fraud_score":0.4}"#;
const BODY_SCORE_06: &[u8] = br#"{"approved":false,"fraud_score":0.6}"#;
const BODY_SCORE_08: &[u8] = br#"{"approved":false,"fraud_score":0.8}"#;
const BODY_SCORE_1: &[u8] = br#"{"approved":false,"fraud_score":1.0}"#;
const RESPONSE_SCORE_0: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}";
const RESPONSE_SCORE_02: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}";
const RESPONSE_SCORE_04: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}";
const RESPONSE_SCORE_06: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}";
const RESPONSE_SCORE_08: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}";
const RESPONSE_SCORE_1: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}";

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
    socket: &'static str,
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
    body: Vec<u8>,
}

enum ParsedRequest {
    Ready,
    FraudScore(Vec<u8>),
    NotFound,
    BadRequest,
    Incomplete,
}

fn main() -> io::Result<()> {
    let address = BIND
        .parse::<SocketAddr>()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;

    let mut poll = Poll::new()?;
    let mut listener = TcpListener::bind(address)?;
    poll.registry()
        .register(&mut listener, SERVER, Interest::READABLE)?;

    let mut lanes = connect_lanes(poll.registry(), UPSTREAM_SOCKETS, CONNECTIONS_PER_UPSTREAM)?;
    let mut idle_lanes = (0..lanes.len()).rev().collect::<Vec<_>>();
    let client_start = UPSTREAM_START + lanes.len();
    let mut pending_jobs = VecDeque::new();
    let mut events = Events::with_capacity(1024);
    let mut connections = Vec::new();
    let mut next_token = client_start;
    println!("LB: {BIND}");

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
    sockets: [&'static str; 2],
    lane_count_per_upstream: usize,
) -> io::Result<Vec<UpstreamLane>> {
    let mut lanes = Vec::with_capacity(sockets.len() * lane_count_per_upstream);

    for socket in sockets {
        for _ in 0..lane_count_per_upstream {
            let token = Token(UPSTREAM_START + lanes.len());
            let mut stream = connect_upstream(&socket)?;
            registry.register(&mut stream, token, Interest::READABLE)?;
            lanes.push(UpstreamLane {
                stream,
                socket,
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
            ParsedRequest::NotFound | ParsedRequest::BadRequest => {
                queue_response(registry, token, connections, ERROR_RESPONSE, true)?
            }
            ParsedRequest::Incomplete => return update_interest(registry, token, connections),
            ParsedRequest::FraudScore(body) => {
                if let Some(connection) = connection_mut(connections, token) {
                    connection.pending = true;
                }
                assign_job(
                    registry,
                    lanes,
                    idle_lanes,
                    pending_jobs,
                    Job {
                        client: token,
                        body,
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
    job: Job,
) -> io::Result<()> {
    let Some(index) = idle_lanes.pop() else {
        pending_jobs.push_back(job);
        return Ok(());
    };

    start_lane_job(registry, &mut lanes[index], job)
}

fn start_lane_job(registry: &mio::Registry, lane: &mut UpstreamLane, job: Job) -> io::Result<()> {
    lane.client = Some(job.client);
    lane.write_buffer.clear();
    lane.write_buffer.push(controller::FRAUD_SCORE_METHOD);
    lane.write_buffer
        .extend_from_slice(&(job.body.len() as u32).to_be_bytes());
    lane.write_buffer.extend_from_slice(&job.body);
    lane.write_offset = 0;
    lane.header = [0; RESPONSE_HEADER_LEN];
    lane.header_offset = 0;
    lane.response_offset = 0;
    lane.response_len = 0;
    lane.status = UmbralStatus::Ok;
    lane.state = LaneState::Writing;
    registry.reregister(&mut lane.stream, lane.token, Interest::WRITABLE)?;
    flush_lane_write(registry, lane)
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
            complete_client_payload(
                registry,
                client,
                connections,
                &lanes[index].response[..lanes[index].response_len],
            )?;
        } else {
            complete_client_bytes(registry, client, connections, ERROR_RESPONSE, true)?;
        }
    }

    lanes[index].state = LaneState::Idle;
    registry.reregister(
        &mut lanes[index].stream,
        lanes[index].token,
        Interest::READABLE,
    )?;

    if let Some(job) = pending_jobs.pop_front() {
        start_lane_job(registry, &mut lanes[index], job)?;
    } else {
        idle_lanes.push(index);
    }

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
        complete_client_bytes(registry, client, connections, ERROR_RESPONSE, true)?;
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

fn complete_client_payload(
    registry: &mio::Registry,
    token: Token,
    connections: &mut Connections,
    payload: &[u8],
) -> io::Result<()> {
    let response = http_response_for_payload(payload).unwrap_or(ERROR_RESPONSE);
    complete_client_bytes(
        registry,
        token,
        connections,
        response,
        response == ERROR_RESPONSE,
    )
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

fn http_response_for_payload(payload: &[u8]) -> Option<&'static [u8]> {
    if payload == BODY_SCORE_0 {
        Some(RESPONSE_SCORE_0)
    } else if payload == BODY_SCORE_02 {
        Some(RESPONSE_SCORE_02)
    } else if payload == BODY_SCORE_04 {
        Some(RESPONSE_SCORE_04)
    } else if payload == BODY_SCORE_06 {
        Some(RESPONSE_SCORE_06)
    } else if payload == BODY_SCORE_08 {
        Some(RESPONSE_SCORE_08)
    } else if payload == BODY_SCORE_1 {
        Some(RESPONSE_SCORE_1)
    } else {
        None
    }
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
    let body_start = header_end + 4;

    if header.starts_with(b"GET /ready ") {
        drain_request(buffer, body_start);
        Ok(ParsedRequest::Ready)
    } else if header.starts_with(b"POST /fraud-score ") {
        let Some(content_length) = find_content_length(header)? else {
            buffer.clear();
            return Ok(ParsedRequest::BadRequest);
        };
        if content_length > MAX_BODY_LEN {
            buffer.clear();
            return Ok(ParsedRequest::BadRequest);
        }

        let total_len = body_start + content_length;
        if buffer.len() < total_len {
            return Ok(ParsedRequest::Incomplete);
        }

        let body = buffer[header_end + 4..total_len].to_vec();
        drain_request(buffer, total_len);
        Ok(ParsedRequest::FraudScore(body))
    } else {
        drain_request(buffer, body_start);
        Ok(ParsedRequest::NotFound)
    }
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

fn find_content_length(header: &[u8]) -> io::Result<Option<usize>> {
    let Some(index) = find_bytes(header, b"\r\nContent-Length:") else {
        return Ok(None);
    };
    let mut value = &header[index + 17..];
    while matches!(value.first(), Some(b' ' | b'\t')) {
        value = &value[1..];
    }
    let line_end = value
        .windows(2)
        .position(|window| window == b"\r\n")
        .unwrap_or(value.len());
    parse_usize(&value[..line_end]).map(Some)
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
    find_bytes(buffer, b"\r\n\r\n")
}

fn find_bytes(buffer: &[u8], needle: &[u8]) -> Option<usize> {
    buffer
        .windows(needle.len())
        .position(|window| window == needle)
}

fn drain_request(buffer: &mut Vec<u8>, end: usize) {
    if end == buffer.len() {
        buffer.clear();
    } else {
        buffer.drain(..end);
    }
}
