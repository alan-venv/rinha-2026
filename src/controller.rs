use actix_web::web::Bytes;
use actix_web::{HttpResponse, Responder, get, post};

#[get("/ready")]
pub async fn ready() -> impl Responder {
    HttpResponse::Ok().body("I was born ready")
}

#[post("/fraud-score")]
// recebe references por "Data" do actix via parametro aqui
pub async fn score() -> HttpResponse {
    // let vector = encoding::vectorization(request.0);
    // let fraud_score = service::fraud_score(&vector); // ,references.as_ref()

    let fraud_score = 0.0;

    HttpResponse::Ok()
        .content_type("application/json")
        .body(response(fraud_score))
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
