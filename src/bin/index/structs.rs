use rinha::*;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct ReferenceJson {
    pub vector: [f32; VECTOR_DIMENSIONS],
    pub label: JsonLabel,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JsonLabel {
    Legit,
    Fraud,
}

pub struct Assignments {
    pub offsets: Vec<u32>,
    pub by_position: Vec<u32>,
}

pub struct AssignmentChunk {
    pub start: usize,
    pub assignments: Vec<u32>,
    pub centroid_counts: Vec<u32>,
}
