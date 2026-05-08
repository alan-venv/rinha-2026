use actix_web::web::{Data, Json};
use actix_web::{HttpResponse, Responder, get, post};

use crate::dto::{ContentRequest, ContentResponse};
use crate::encoding;
use crate::memory::IndexedReferences;
use crate::service;

#[get("/ready")]
pub async fn ready() -> impl Responder {
    return HttpResponse::Ok().body("I was born ready");
}

#[post("/fraud-score")]
pub async fn score(
    request: Json<ContentRequest>,
    references: Data<IndexedReferences>,
) -> impl Responder {
    let vector = encoding::vectorization(request.0);
    let score = service::fraud_score(&vector, references.as_ref());

    let response = ContentResponse::from(score);
    return HttpResponse::Ok().json(response);
}
