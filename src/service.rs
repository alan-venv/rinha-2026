use crate::morton::MortonIndex;

pub fn fraud_score(vector: &[i16; 14], index: &MortonIndex) -> f32 {
    index.fraud_score(vector)
}
