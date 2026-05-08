use std::cell::Cell;

use crate::memory::ReferenceSource;
use crate::*;

#[allow(dead_code)]
pub fn fraud_score(vector: &ReferenceVector, records: &(impl ReferenceSource + ?Sized)) -> f32 {
    fraud_score_details(vector, records).score
}

fn score_from_nearest_result(
    nearest_result: &NearestResult,
    records: &(impl ReferenceSource + ?Sized),
) -> f32 {
    let candidates = nearest_result.candidates();
    let score_candidates = &candidates[..candidates.len().min(NEAREST_COUNT)];
    let top_fraud_count = fraud_count(score_candidates, records);

    top_fraud_count as f32 / score_candidates.len() as f32
}

fn fraud_count(nearest: &[NearestCandidate], records: &(impl ReferenceSource + ?Sized)) -> usize {
    nearest
        .iter()
        .filter(|candidate| records.is_fraud(candidate.index))
        .count()
}

#[derive(Clone, Copy)]
struct NearestCandidate {
    index: usize,
    distance: u64,
}

struct NearestResult {
    candidates: [NearestCandidate; NEAREST_COUNT],
    len: usize,
    boundary_case: bool,
    probes_used: usize,
    fallback_used: bool,
    rescue_used: bool,
}

impl NearestResult {
    fn candidates(&self) -> &[NearestCandidate] {
        &self.candidates[..self.len]
    }
}

struct TopNearest {
    candidates: [NearestCandidate; NEAREST_COUNT],
    len: usize,
    farthest_slot: usize,
}

impl TopNearest {
    fn new() -> Self {
        Self {
            candidates: [NearestCandidate {
                index: 0,
                distance: u64::MAX,
            }; NEAREST_COUNT],
            len: 0,
            farthest_slot: 0,
        }
    }

    fn current_worst_distance(&self) -> u64 {
        if self.len < NEAREST_COUNT {
            u64::MAX
        } else {
            self.candidates[self.farthest_slot].distance
        }
    }

    fn add(&mut self, candidate: NearestCandidate) {
        if self.len == NEAREST_COUNT && candidate.distance >= self.current_worst_distance() {
            return;
        }

        if self.contains(candidate.index) {
            return;
        }

        if self.len < NEAREST_COUNT {
            self.candidates[self.len] = candidate;
            self.len += 1;

            if candidate.distance > self.candidates[self.farthest_slot].distance {
                self.farthest_slot = self.len - 1;
            }

            return;
        }

        self.candidates[self.farthest_slot] = candidate;
        self.farthest_slot = self.find_farthest_slot();
    }

    fn sort_candidates(&mut self) {
        self.candidates[..self.len].sort_unstable_by_key(|candidate| candidate.distance);
        self.farthest_slot = self.find_farthest_slot();
    }

    fn contains(&self, index: usize) -> bool {
        self.candidates[..self.len]
            .iter()
            .any(|candidate| candidate.index == index)
    }

    fn find_farthest_slot(&self) -> usize {
        let mut farthest_slot = 0;

        for slot in 1..self.len {
            if self.candidates[slot].distance > self.candidates[farthest_slot].distance {
                farthest_slot = slot;
            }
        }

        farthest_slot
    }
}

fn nearest(vector: &ReferenceVector, records: &(impl ReferenceSource + ?Sized)) -> NearestResult {
    let context = records.prepare_search_context(vector);
    let mut nearest = TopNearest::new();
    let current_worst_distance = Cell::new(u64::MAX);
    records.for_each_primary_candidate_batch(
        &context,
        vector,
        0,
        IVF_INITIAL_PROBES,
        &mut || current_worst_distance.get(),
        &mut |index, distance| {
            nearest.add(NearestCandidate { index, distance });
            current_worst_distance.set(nearest.current_worst_distance());
        },
    );

    nearest.sort_candidates();
    let boundary_case = is_boundary_case(&nearest.candidates[..nearest.len], records);

    if !boundary_case {
        return NearestResult {
            candidates: nearest.candidates,
            len: nearest.len,
            boundary_case,
            probes_used: IVF_INITIAL_PROBES,
            fallback_used: false,
            rescue_used: false,
        };
    }

    let mut fallback_nearest = TopNearest::new();
    let fallback_current_worst_distance = Cell::new(u64::MAX);
    let fallback_used = records.for_each_boundary_fallback_candidate(
        &context,
        vector,
        &mut || fallback_current_worst_distance.get(),
        &mut |index, distance| {
            fallback_nearest.add(NearestCandidate { index, distance });
            fallback_current_worst_distance.set(fallback_nearest.current_worst_distance());
        },
    );

    if fallback_used {
        fallback_nearest.sort_candidates();
        let fallback_candidates = &fallback_nearest.candidates[..fallback_nearest.len];
        let fallback_score_candidates =
            &fallback_candidates[..fallback_candidates.len().min(NEAREST_COUNT)];
        let fallback_fraud_count = fraud_count(fallback_score_candidates, records);
        let boundary_case = is_boundary_case(fallback_candidates, records);

        if fallback_fraud_count == 3 {
            let mut rescue_nearest = TopNearest::new();
            let rescue_current_worst_distance = Cell::new(u64::MAX);
            let rescue_used = records.for_each_boundary_rescue_candidate(
                &context,
                vector,
                &mut || rescue_current_worst_distance.get(),
                &mut |index, distance| {
                    rescue_nearest.add(NearestCandidate { index, distance });
                    rescue_current_worst_distance.set(rescue_nearest.current_worst_distance());
                },
            );

            if rescue_used {
                rescue_nearest.sort_candidates();
                let boundary_case =
                    is_boundary_case(&rescue_nearest.candidates[..rescue_nearest.len], records);

                return NearestResult {
                    candidates: rescue_nearest.candidates,
                    len: rescue_nearest.len,
                    boundary_case,
                    probes_used: records.boundary_rescue_probe_count(),
                    fallback_used: true,
                    rescue_used: true,
                };
            }
        }

        return NearestResult {
            candidates: fallback_nearest.candidates,
            len: fallback_nearest.len,
            boundary_case,
            probes_used: records.boundary_fallback_probe_count(),
            fallback_used: true,
            rescue_used: false,
        };
    }

    NearestResult {
        candidates: nearest.candidates,
        len: nearest.len,
        boundary_case,
        probes_used: IVF_INITIAL_PROBES,
        fallback_used: false,
        rescue_used: false,
    }
}

fn is_boundary_case(
    candidates: &[NearestCandidate],
    records: &(impl ReferenceSource + ?Sized),
) -> bool {
    if candidates.len() < NEAREST_COUNT {
        return true;
    }

    let top = &candidates[..candidates.len().min(NEAREST_COUNT)];
    let frauds = fraud_count(top, records);

    frauds > 0 && frauds < top.len()
}

pub struct FraudScoreDetails {
    pub score: f32,
    #[allow(dead_code)]
    pub boundary_case: bool,
    #[allow(dead_code)]
    pub probes_used: usize,
    #[allow(dead_code)]
    pub fallback_used: bool,
    #[allow(dead_code)]
    pub rescue_used: bool,
}

pub fn fraud_score_details(
    vector: &ReferenceVector,
    records: &(impl ReferenceSource + ?Sized),
) -> FraudScoreDetails {
    let nearest_result = nearest(vector, records);

    FraudScoreDetails {
        score: score_from_nearest_result(&nearest_result, records),
        boundary_case: nearest_result.boundary_case,
        probes_used: nearest_result.probes_used,
        fallback_used: nearest_result.fallback_used,
        rescue_used: nearest_result.rescue_used,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{ReferenceRecord, ReferenceSource};
    fn record(value: i16, is_fraud: bool) -> ReferenceRecord {
        ReferenceRecord {
            vector: [value; VECTOR_DIMENSIONS],
            is_fraud,
        }
    }

    fn exact_distance2(a: &ReferenceVector, b: &ReferenceVector) -> u64 {
        a.iter()
            .zip(b)
            .map(|(left, right)| {
                let delta = i64::from(*left) - i64::from(*right);
                (delta * delta) as u64
            })
            .sum()
    }

    fn euclidean_distance2(a: &ReferenceVector, b: &ReferenceVector) -> u64 {
        exact_distance2(a, b)
    }

    struct GroupedRecords {
        frauds: [bool; 10],
        groups: [[(usize, u64); 5]; 2],
    }

    impl ReferenceSource for GroupedRecords {
        fn is_fraud(&self, index: usize) -> bool {
            self.frauds[index]
        }

        fn for_each_primary_candidate_batch<C, V>(
            &self,
            _context: &crate::memory::SearchContext,
            _vector: &ReferenceVector,
            start_probe: usize,
            _end_probe: usize,
            _current_worst_top_distance: &mut C,
            visit: &mut V,
        ) where
            C: FnMut() -> u64,
            V: FnMut(usize, u64),
        {
            if start_probe > 0 {
                return;
            }

            for (index, distance) in &self.groups[0] {
                visit(*index, *distance);
            }
        }
    }

    #[test]
    fn calculates_euclidean_distance_without_square_root() {
        let mut a = [0; VECTOR_DIMENSIONS];
        let mut b = [0; VECTOR_DIMENSIONS];
        a[0] = 1_000;
        b[0] = 4_000;
        a[1] = 2_000;
        b[1] = 6_000;

        assert_eq!(euclidean_distance2(&a, &b), 25_000_000);
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
            nearest(&query, &records)
                .candidates()
                .iter()
                .take(NEAREST_COUNT)
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
            record(6_000, false),
            record(7_000, true),
            record(8_000, false),
            record(9_000, true),
            record(10_000, false),
            record(20_000, true),
        ];

        assert_eq!(fraud_score(&query, &records), 0.4);
    }

    #[test]
    fn marks_boundary_case_when_score_is_zero_point_six() {
        let query = [0; VECTOR_DIMENSIONS];
        let records = GroupedRecords {
            frauds: [
                true, true, true, false, false, true, false, false, false, false,
            ],
            groups: [
                [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
                [(5, 1), (6, 10), (7, 11), (8, 12), (9, 13)],
            ],
        };

        let details = fraud_score_details(&query, &records);

        assert_eq!(details.score, 0.6);
        assert!(details.boundary_case);
    }

    #[test]
    fn marks_boundary_case_when_score_is_zero_point_four() {
        let query = [0; VECTOR_DIMENSIONS];
        let records = GroupedRecords {
            frauds: [
                true, true, false, false, false, true, true, true, true, true,
            ],
            groups: [
                [(0, 0), (1, 1), (2, 2), (3, 3), (4, 500_000)],
                [(5, 0), (6, 1), (7, 2), (8, 3), (9, 4)],
            ],
        };

        let details = fraud_score_details(&query, &records);

        assert_eq!(details.score, 0.4);
        assert!(details.boundary_case);
    }

    #[test]
    fn marks_boundary_case_when_score_is_zero_point_two() {
        let query = [0; VECTOR_DIMENSIONS];
        let records = GroupedRecords {
            frauds: [
                true, false, false, false, false, true, true, true, true, true,
            ],
            groups: [
                [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
                [(5, 0), (6, 1), (7, 2), (8, 3), (9, 4)],
            ],
        };

        let details = fraud_score_details(&query, &records);

        assert_eq!(details.score, 0.2);
        assert!(details.boundary_case);
    }

    #[test]
    fn marks_boundary_case_when_score_is_zero_point_eight() {
        let query = [0; VECTOR_DIMENSIONS];
        let records = GroupedRecords {
            frauds: [
                true, true, true, true, false, false, false, false, false, false,
            ],
            groups: [
                [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
                [(5, 0), (6, 1), (7, 2), (8, 3), (9, 4)],
            ],
        };

        let details = fraud_score_details(&query, &records);

        assert_eq!(details.score, 0.8);
        assert!(details.boundary_case);
    }

    #[test]
    fn does_not_mark_boundary_case_when_score_is_pure() {
        let query = [0; VECTOR_DIMENSIONS];
        let records = GroupedRecords {
            frauds: [
                false, false, false, false, false, false, false, false, false, false,
            ],
            groups: [
                [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
                [(5, 5), (6, 6), (7, 7), (8, 8), (9, 9)],
            ],
        };

        let details = fraud_score_details(&query, &records);

        assert_eq!(details.score, 0.0);
        assert!(!details.boundary_case);
    }

    #[test]
    fn does_not_mark_boundary_case_from_mixed_neighbors_after_top_5() {
        let query = [0; VECTOR_DIMENSIONS];
        let records = [
            record(0, false),
            record(0, false),
            record(0, false),
            record(0, false),
            record(0, false),
            record(0, true),
        ];

        let details = fraud_score_details(&query, &records);

        assert_eq!(details.score, 0.0);
        assert!(!details.boundary_case);
    }

    #[test]
    fn ignores_duplicate_candidates_in_primary_results() {
        let query = [0; VECTOR_DIMENSIONS];
        let records = GroupedRecords {
            frauds: [
                true, false, false, false, false, true, true, true, true, true,
            ],
            groups: [
                [(0, 0), (0, 0), (1, 1), (2, 2), (3, 3)],
                [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8)],
            ],
        };

        assert_eq!(
            nearest(&query, &records)
                .candidates()
                .iter()
                .map(|candidate| candidate.index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }
}
