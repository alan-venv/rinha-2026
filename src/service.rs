use std::cell::Cell;

use chrono::{Datelike, Timelike};

use crate::consts::*;
use crate::dto::{ContentRequest, Customer, LastTransaction, Merchant, Transaction};
use crate::memory::ReferenceSource;
use rinha::*;

pub struct FraudScoreDetails {
    pub score: f32,
    #[allow(dead_code)]
    pub boundary_case: bool,
    #[allow(dead_code)]
    pub probes_used: usize,
    #[allow(dead_code)]
    pub fallback_used: bool,
}

#[allow(dead_code)]
pub fn fraud_score(vector: &ReferenceVector, records: &(impl ReferenceSource + ?Sized)) -> f32 {
    fraud_score_details(vector, records).score
}

pub fn fraud_score_details(
    vector: &ReferenceVector,
    records: &(impl ReferenceSource + ?Sized),
) -> FraudScoreDetails {
    let nearest_result = nearest(vector, records);
    let candidates = nearest_result.candidates();
    let score_candidates = &candidates[..candidates.len().min(NEAREST_COUNT)];
    let top_fraud_count = fraud_count(score_candidates, records);

    FraudScoreDetails {
        score: top_fraud_count as f32 / score_candidates.len() as f32,
        boundary_case: nearest_result.boundary_case,
        probes_used: nearest_result.probes_used,
        fallback_used: nearest_result.fallback_used,
    }
}

fn fraud_count(nearest: &[NearestCandidate], records: &(impl ReferenceSource + ?Sized)) -> usize {
    nearest
        .iter()
        .filter(|candidate| records.is_fraud(candidate.index))
        .count()
}

#[cfg(test)]
fn exact_distance2(a: &ReferenceVector, b: &ReferenceVector) -> u64 {
    a.iter()
        .zip(b)
        .map(|(left, right)| {
            let delta = i64::from(*left) - i64::from(*right);
            (delta * delta) as u64
        })
        .sum()
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
    let mut nearest = TopNearest::new();
    let current_worst_distance = Cell::new(u64::MAX);
    records.for_each_primary_candidate_batch(
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
        };
    }

    let mut fallback_nearest = TopNearest::new();
    let fallback_current_worst_distance = Cell::new(u64::MAX);
    let fallback_used = records.for_each_boundary_fallback_candidate(
        vector,
        &mut || fallback_current_worst_distance.get(),
        &mut |index, distance| {
            fallback_nearest.add(NearestCandidate { index, distance });
            fallback_current_worst_distance.set(fallback_nearest.current_worst_distance());
        },
    );

    if fallback_used {
        fallback_nearest.sort_candidates();
        let boundary_case = is_boundary_case(
            &fallback_nearest.candidates[..fallback_nearest.len],
            records,
        );

        return NearestResult {
            candidates: fallback_nearest.candidates,
            len: fallback_nearest.len,
            boundary_case,
            probes_used: BOUNDARY_FULL_SCAN_FINE_PROBES,
            fallback_used: true,
        };
    }

    NearestResult {
        candidates: nearest.candidates,
        len: nearest.len,
        boundary_case,
        probes_used: IVF_INITIAL_PROBES,
        fallback_used: false,
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

#[cfg(test)]
fn euclidean_distance2(a: &ReferenceVector, b: &ReferenceVector) -> u64 {
    exact_distance2(a, b)
}

pub fn vectorization(request: ContentRequest) -> ReferenceVector {
    let transaction = &request.transaction;
    let last_transaction = &request.last_transaction;
    let customer = &request.customer;
    let merchant = &request.merchant;
    let terminal = &request.terminal;

    let n0 = clamp(transaction.amount / MAX_AMOUNT);
    let n1 = clamp(transaction.installments / MAX_INSTALLMENTS);
    let n2 = clamp((transaction.amount / customer.avg_amount) / AMOUNT_VS_AVG_RATIO);
    let n3 = transaction.requested_at.hour() as f32 / 23.0;
    let n4 = transaction.requested_at.weekday().num_days_from_monday() as f32 / 6.0;
    let n5 = minutes_since_last_transaction(transaction, last_transaction);
    let n6 = km_from_last_transaction(last_transaction);
    let n7 = clamp(terminal.km_from_home / MAX_KM);
    let n8 = clamp(customer.tx_count_24h / MAX_TX_COUNT_24H);
    let n9: f32 = if terminal.is_online { 1.0 } else { 0.0 };
    let n10: f32 = if terminal.card_present { 1.0 } else { 0.0 };
    let n11 = unknown_merchant(customer, merchant);
    let n12 = mcc_risk(&merchant.mcc);
    let n13 = clamp(merchant.avg_amount / MAX_MERCHANT_AVG_AMOUNT);

    [
        quantize(n0),
        quantize(n1),
        quantize(n2),
        quantize(n3),
        quantize(n4),
        quantize(n5),
        quantize(n6),
        quantize(n7),
        quantize(n8),
        quantize(n9),
        quantize(n10),
        quantize(n11),
        quantize(n12),
        quantize(n13),
    ]
}

fn clamp(value: f64) -> f32 {
    value.clamp(0.0, 1.0) as f32
}

fn quantize(value: f32) -> i16 {
    (value * VECTOR_SCALE).round() as i16
}

fn minutes_since_last_transaction(
    transaction: &Transaction,
    last_transaction: &Option<LastTransaction>,
) -> f32 {
    let requested_at = &transaction.requested_at;
    last_transaction.as_ref().map_or(-1.0, |last| {
        let minutes = requested_at
            .signed_duration_since(last.timestamp)
            .num_minutes()
            .max(0) as f64
            / MAX_MINUTES;
        clamp(minutes)
    })
}

fn km_from_last_transaction(last_transaction: &Option<LastTransaction>) -> f32 {
    last_transaction
        .as_ref()
        .map_or(-1.0, |last| clamp(last.km_from_current / MAX_KM))
}

fn unknown_merchant(customer: &Customer, merchant: &Merchant) -> f32 {
    if customer.known_merchants.contains(&merchant.id) {
        0.0
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{ReferenceRecord, ReferenceSource};
    use chrono::{DateTime, TimeZone, Utc};

    fn utc(year: i32, month: u32, day: u32, hour: u32, min: u32, sec: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(year, month, day, hour, min, sec)
            .unwrap()
    }

    fn transaction(requested_at: DateTime<Utc>) -> Transaction {
        Transaction {
            amount: 0.0,
            installments: 0.0,
            requested_at,
        }
    }

    fn customer(known_merchants: Vec<String>) -> Customer {
        Customer {
            avg_amount: 0.0,
            tx_count_24h: 0.0,
            known_merchants,
        }
    }

    fn merchant(id: &str) -> Merchant {
        Merchant {
            id: id.to_string(),
            mcc: String::new(),
            avg_amount: 0.0,
        }
    }

    fn record(value: i16, is_fraud: bool) -> ReferenceRecord {
        ReferenceRecord {
            vector: [value; VECTOR_DIMENSIONS],
            is_fraud,
        }
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
    fn clamps_values_below_zero() {
        assert_eq!(clamp(-0.1), 0.0);
        assert_eq!(clamp(-10.0), 0.0);
    }

    #[test]
    fn clamps_values_above_one() {
        assert_eq!(clamp(1.1), 1.0);
        assert_eq!(clamp(10.0), 1.0);
    }

    #[test]
    fn keeps_values_between_zero_and_one() {
        assert_eq!(clamp(0.0), 0.0);
        assert_eq!(clamp(0.5), 0.5);
        assert_eq!(clamp(1.0), 1.0);
    }

    #[test]
    fn quantizes_to_four_decimal_places() {
        assert_eq!(quantize(0.038488), 385);
        assert_eq!(quantize(0.33333334), 3333);
        assert_eq!(quantize(1.0), 10_000);
        assert_eq!(quantize(-1.0), -10_000);
    }

    #[test]
    fn calculates_normalized_minutes_since_last_transaction() {
        let transaction = transaction(utc(2026, 3, 11, 20, 23, 35));
        let last_transaction = LastTransaction {
            timestamp: utc(2026, 3, 11, 14, 58, 35),
            km_from_current: 0.0,
        };

        assert_eq!(
            minutes_since_last_transaction(&transaction, &Some(last_transaction)),
            (325.0 / MAX_MINUTES) as f32
        );
    }

    #[test]
    fn returns_minus_one_when_last_transaction_is_missing() {
        let transaction = transaction(utc(2026, 3, 11, 20, 23, 35));

        assert_eq!(minutes_since_last_transaction(&transaction, &None), -1.0);
    }

    #[test]
    fn does_not_return_negative_minutes() {
        let transaction = transaction(utc(2026, 3, 11, 20, 23, 35));
        let last_transaction = LastTransaction {
            timestamp: utc(2026, 3, 11, 21, 23, 35),
            km_from_current: 0.0,
        };

        assert_eq!(
            minutes_since_last_transaction(&transaction, &Some(last_transaction)),
            0.0
        );
    }

    #[test]
    fn calculates_normalized_km_from_last_transaction() {
        let last_transaction = LastTransaction {
            timestamp: utc(2026, 3, 11, 14, 58, 35),
            km_from_current: 250.0,
        };

        assert_eq!(km_from_last_transaction(&Some(last_transaction)), 0.25);
    }

    #[test]
    fn clamps_km_from_last_transaction() {
        let last_transaction = LastTransaction {
            timestamp: utc(2026, 3, 11, 14, 58, 35),
            km_from_current: 2_000.0,
        };

        assert_eq!(km_from_last_transaction(&Some(last_transaction)), 1.0);
    }

    #[test]
    fn returns_minus_one_when_last_transaction_km_is_missing() {
        assert_eq!(km_from_last_transaction(&None), -1.0);
    }

    #[test]
    fn returns_zero_for_known_merchant() {
        assert_eq!(
            unknown_merchant(
                &customer(vec!["MERC-001".to_string()]),
                &merchant("MERC-001")
            ),
            0.0
        );
    }

    #[test]
    fn returns_one_for_unknown_merchant() {
        assert_eq!(
            unknown_merchant(
                &customer(vec!["MERC-001".to_string()]),
                &merchant("MERC-002")
            ),
            1.0
        );
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
