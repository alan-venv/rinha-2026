use std::io;

use crate::encoding;
use crate::parser;
use crate::service::Service;

pub fn ready_handler(_: &Service, _: &[u8]) -> io::Result<&'static [u8]> {
    Ok(b"HTTP/1.1 200 OK\r\nContent-Length: 16\r\nConnection: keep-alive\r\n\r\nI was born ready")
}

pub fn score_handler(service: &Service, body: &[u8]) -> io::Result<&'static [u8]> {
    let request = parser::parse(body)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid request"))?;
    let vector = encoding::vectorization(&request);
    let score = service.fraud_score(&vector);

    Ok(score_response(score))
}

fn score_response(score: f32) -> &'static [u8] {
    const RESPONSES: [&[u8]; 6] = [
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
    ];
    RESPONSES[(score * 5.0).round() as usize]
}
