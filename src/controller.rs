use std::sync::Arc;

use ntex::web::types::{Json, State};
use ntex::web::{self, HttpResponse, Responder};

use crate::dto::{ContentRequest, ContentResponse};
use crate::memory::IndexedReferences;
use crate::service;

#[web::get("/ready")]
pub async fn ready() -> impl Responder {
    return HttpResponse::Ok().body("I was born ready");
}

#[web::post("/fraud-score")]
pub async fn score(
    request: Json<ContentRequest>,
    references: State<Arc<IndexedReferences>>,
) -> impl Responder {
    let vector = service::vectorization(request.into_inner());
    let score = service::fraud_score(&vector, references.get_ref().as_ref());

    let response = ContentResponse::from(score);
    return HttpResponse::Ok().json(&response);
}
