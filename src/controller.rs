use actix_web::web::Bytes;
use actix_web::web::Data;
use actix_web::{HttpResponse, Responder, get, post};

use crate::encoding;
use crate::parser;
use crate::service::Service;

#[get("/ready")]
pub async fn ready() -> impl Responder {
    HttpResponse::Ok().body("I was born ready")
}

#[post("/fraud-score")]
pub async fn score(body: Bytes, service: Data<Service>) -> HttpResponse {
    let Ok(request) = parser::parse(&body) else {
        return HttpResponse::BadRequest().finish();
    };
    let vector = encoding::vectorization(&request);
    let score = service.fraud_score(&vector);

    HttpResponse::Ok()
        .content_type("application/json")
        .body(response(score))
}

fn response(fraud_score: f32) -> Bytes {
    const RESPONSES: [&[u8]; 6] = [
        br#"{"approved":true,"fraud_score":0.0}"#,
        br#"{"approved":true,"fraud_score":0.2}"#,
        br#"{"approved":true,"fraud_score":0.4}"#,
        br#"{"approved":false,"fraud_score":0.6}"#,
        br#"{"approved":false,"fraud_score":0.8}"#,
        br#"{"approved":false,"fraud_score":1.0}"#,
    ];
    Bytes::from_static(RESPONSES[(fraud_score * 5.0).round() as usize])
}
