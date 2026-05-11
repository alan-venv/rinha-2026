use actix_web::web::{Bytes, Json};
use actix_web::{HttpResponse, Responder, get, post};

use crate::dto::ContentRequest;
use crate::encoding;
use crate::service;

#[get("/ready")]
pub async fn ready() -> impl Responder {
    HttpResponse::Ok().body("I was born ready")
}

#[post("/fraud-score")]
// recebe references por "Data" do actix via parametro aqui
pub async fn score(request: Json<ContentRequest>) -> HttpResponse {
    let vector = encoding::vectorization(request.0);
    let fraud_score = service::fraud_score(&vector); // ,references.as_ref()

    HttpResponse::Ok()
        .content_type("application/json")
        .body(Bytes::from_static(static_response_body(fraud_score)))
}

fn static_response_body(fraud_score: f32) -> &'static [u8] {
    SCORE_RESPONSES[(fraud_score * 5.0).round() as usize]
}

const SCORE_RESPONSES: [&[u8]; 6] = [
    br#"{"approved":true,"fraud_score":0.0}"#,
    br#"{"approved":true,"fraud_score":0.2}"#,
    br#"{"approved":true,"fraud_score":0.4}"#,
    br#"{"approved":false,"fraud_score":0.6}"#,
    br#"{"approved":false,"fraud_score":0.8}"#,
    br#"{"approved":false,"fraud_score":1.0}"#,
];
