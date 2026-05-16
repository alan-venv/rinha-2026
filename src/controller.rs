use std::io;

use crate::encoding;
use crate::parser;
use crate::response::fraud_response;
use crate::service::Service;
use umbral_socket::stream::UmbralResponse;

pub const READY_RESPONSE: &[u8] = b"I was born ready";
pub const FRAUD_SCORE_METHOD: u8 = 1;
pub const READY_METHOD: u8 = 2;

pub fn ready() -> &'static [u8] {
    READY_RESPONSE
}

pub fn score(service: &Service, body: &[u8]) -> io::Result<&'static [u8]> {
    let request = parser::parse(body)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid request"))?;
    let vector = encoding::vectorization(&request);
    let score = service.fraud_score(&vector);

    Ok(fraud_response(score))
}

pub fn score_handler(
    service: &Service,
    payload: &[u8],
    _: &mut Vec<u8>,
) -> io::Result<UmbralResponse> {
    Ok(UmbralResponse::Static(score(service, payload)?))
}

pub fn ready_handler(_: &Service, _: &[u8], _: &mut Vec<u8>) -> io::Result<UmbralResponse> {
    Ok(UmbralResponse::Static(ready()))
}
