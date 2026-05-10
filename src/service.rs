use crate::memory::{NearestCandidate, ReferenceSource};
use crate::*;

pub fn fraud_score(vector: &ReferenceVector, records: &(impl ReferenceSource + ?Sized)) -> f32 {
    fraud_score_details(vector, records).score
}

pub struct FraudScoreDetails {
    pub score: f32,
    pub boundary_case: bool,
}

pub fn fraud_score_details(
    vector: &ReferenceVector,
    records: &(impl ReferenceSource + ?Sized),
) -> FraudScoreDetails {
    let nearest = records.search_top(vector, NEAREST_COUNT);
    let score = score_from_nearest(&nearest, records);

    FraudScoreDetails {
        score,
        boundary_case: is_boundary_case(&nearest, records),
    }
}

fn score_from_nearest(
    nearest: &[NearestCandidate],
    records: &(impl ReferenceSource + ?Sized),
) -> f32 {
    let score_candidates = &nearest[..nearest.len().min(NEAREST_COUNT)];

    if score_candidates.is_empty() {
        return 0.0;
    }

    fraud_count(score_candidates, records) as f32 / score_candidates.len() as f32
}

fn fraud_count(nearest: &[NearestCandidate], records: &(impl ReferenceSource + ?Sized)) -> usize {
    nearest
        .iter()
        .filter(|candidate| records.is_fraud(candidate.index))
        .count()
}

fn is_boundary_case(
    candidates: &[NearestCandidate],
    records: &(impl ReferenceSource + ?Sized),
) -> bool {
    if candidates.len() < NEAREST_COUNT {
        return true;
    }

    let top = &candidates[..NEAREST_COUNT];
    let frauds = fraud_count(top, records);

    frauds > 0 && frauds < top.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ReferenceRecord;

    fn record(value: i16, is_fraud: bool) -> ReferenceRecord {
        let mut vector = [0; VECTOR_DIMENSIONS];
        vector[0] = value;
        ReferenceRecord { vector, is_fraud }
    }

    #[test]
    fn returns_nearest_5() {
        let query = [0; VECTOR_DIMENSIONS];
        let records = [
            record(0, false),
            record(1_000, true),
            record(2_000, false),
            record(3_000, true),
            record(4_000, false),
            record(5_000, true),
            record(6_000, false),
            record(7_000, true),
            record(8_000, false),
            record(9_000, true),
            record(10_000, false),
            record(20_000, true),
        ];

        assert_eq!(
            records
                .search_top(&query, NEAREST_COUNT)
                .iter()
                .map(|candidate| candidate.index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn calculates_fraud_score_from_nearest_5() {
        let query = [0; VECTOR_DIMENSIONS];
        let records = [
            record(0, false),
            record(1_000, true),
            record(2_000, false),
            record(3_000, true),
            record(4_000, false),
            record(5_000, true),
        ];

        assert_eq!(fraud_score(&query, &records), 0.4);
    }

    #[test]
    fn marks_boundary_case_when_score_is_mixed() {
        let query = [0; VECTOR_DIMENSIONS];
        let records = [
            record(0, true),
            record(1_000, true),
            record(2_000, true),
            record(3_000, false),
            record(4_000, false),
        ];

        let details = fraud_score_details(&query, &records);

        assert_eq!(details.score, 0.6);
        assert!(details.boundary_case);
    }

    #[test]
    fn does_not_mark_boundary_case_when_score_is_pure() {
        let query = [0; VECTOR_DIMENSIONS];
        let records = [
            record(0, false),
            record(1_000, false),
            record(2_000, false),
            record(3_000, false),
            record(4_000, false),
        ];

        let details = fraud_score_details(&query, &records);

        assert_eq!(details.score, 0.0);
        assert!(!details.boundary_case);
    }

    #[test]
    fn marks_boundary_case_when_top_is_incomplete() {
        let query = [0; VECTOR_DIMENSIONS];
        let records = [record(0, false)];

        let details = fraud_score_details(&query, &records);

        assert_eq!(details.score, 0.0);
        assert!(details.boundary_case);
    }
}
