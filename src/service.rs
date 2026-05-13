use crate::{kdtree::KdTree, morton::MortonIndex};

pub struct Service {
    morton: MortonIndex,
    kdtree: KdTree,
}

impl Service {
    pub fn new(morton: MortonIndex, kdtree: KdTree) -> Self {
        Self { morton, kdtree }
    }

    pub fn fraud_score(&self, vector: &[i16; 14]) -> f32 {
        self.decide(vector).fraud_score
    }

    pub fn decide(&self, vector: &[i16; 14]) -> FraudDecision {
        if let Some(fraud_score) = self.morton.score(vector) {
            return FraudDecision {
                fraud_score,
                source: DecisionSource::Morton,
            };
        }

        FraudDecision {
            fraud_score: self.kdtree.score(&self.morton, vector),
            source: DecisionSource::KdTree,
        }
    }
}

pub struct FraudDecision {
    pub fraud_score: f32,
    pub source: DecisionSource,
}

pub enum DecisionSource {
    Morton,
    KdTree,
}
