use actix_web::web::{Bytes, Data, Json};
use actix_web::{HttpResponse, Responder, get, post};

use crate::dto::ContentRequest;
use crate::encoding;
use crate::memory::IndexedReferences;
use crate::service;

#[get("/ready")]
pub async fn ready() -> impl Responder {
    HttpResponse::Ok().body("I was born ready")
}

#[post("/fraud-score")]
pub async fn score(
    request: Json<ContentRequest>,
    references: Data<IndexedReferences>,
) -> HttpResponse {
    let vector = encoding::vectorization(request.0);
    let fraud_score = service::fraud_score(&vector, references.as_ref());

    HttpResponse::Ok()
        .content_type("application/json")
        .body(Bytes::from_static(static_response_body(fraud_score)))
}

const SCORE_RESPONSES: [&[u8]; 6] = [
    br#"{"approved":true,"fraud_score":0.0}"#,
    br#"{"approved":true,"fraud_score":0.2}"#,
    br#"{"approved":true,"fraud_score":0.4}"#,
    br#"{"approved":false,"fraud_score":0.6}"#,
    br#"{"approved":false,"fraud_score":0.8}"#,
    br#"{"approved":false,"fraud_score":1.0}"#,
];

fn static_response_body(fraud_score: f32) -> &'static [u8] {
    SCORE_RESPONSES[(fraud_score * 5.0).round() as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(fraud_score: f32) -> &'static str {
        std::str::from_utf8(static_response_body(fraud_score)).unwrap()
    }

    #[test]
    fn static_response_uses_threshold_at_06() {
        assert_eq!(response(0.4), r#"{"approved":true,"fraud_score":0.4}"#);
        assert_eq!(response(0.6), r#"{"approved":false,"fraud_score":0.6}"#);
    }

    #[test]
    fn static_response_covers_all_normal_scores() {
        assert_eq!(response(0.0), r#"{"approved":true,"fraud_score":0.0}"#);
        assert_eq!(response(0.2), r#"{"approved":true,"fraud_score":0.2}"#);
        assert_eq!(response(0.4), r#"{"approved":true,"fraud_score":0.4}"#);
        assert_eq!(response(0.6), r#"{"approved":false,"fraud_score":0.6}"#);
        assert_eq!(response(0.8), r#"{"approved":false,"fraud_score":0.8}"#);
        assert_eq!(response(1.0), r#"{"approved":false,"fraud_score":1.0}"#);
    }
}
