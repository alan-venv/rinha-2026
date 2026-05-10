use serde::Deserialize;

#[derive(Deserialize)]
pub struct ReferenceJson {
    pub vector: [f32; 14],
    pub label: JsonLabel,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JsonLabel {
    Legit,
    Fraud,
}
