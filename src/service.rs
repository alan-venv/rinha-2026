use std::cell::Cell;

use chrono::{Datelike, Timelike};

use crate::consts::*;
use crate::dto::{ContentRequest, Customer, LastTransaction, Merchant, Transaction};
use crate::memory::ReferenceSource;
use rinha::*;

const NEAREST_COUNT: usize = 5;

#[allow(dead_code)]
pub struct FraudScoreDetails {
    pub score: f32,
    pub boundary_case: bool,
}

#[allow(dead_code)]
pub fn fraud_score(vector: &ReferenceVector, records: &(impl ReferenceSource + ?Sized)) -> f32 {
    fraud_score_details(vector, records).score
}

pub fn fraud_score_details(
    vector: &ReferenceVector,
    records: &(impl ReferenceSource + ?Sized),
) -> FraudScoreDetails {
    let nearest = nearest(vector, records, IVF_INITIAL_PROBES);
    let frauds = fraud_count(&nearest.candidates, records);

    FraudScoreDetails {
        score: frauds as f32 / nearest.candidates.len() as f32,
        boundary_case: nearest.boundary_case,
    }
}

fn fraud_count(nearest: &[(usize, u64)], records: &(impl ReferenceSource + ?Sized)) -> usize {
    nearest
        .iter()
        .filter(|(index, _)| records.is_fraud(*index))
        .count()
}

struct NearestResult {
    candidates: Vec<(usize, u64)>,
    boundary_case: bool,
}

fn nearest(
    vector: &ReferenceVector,
    records: &(impl ReferenceSource + ?Sized),
    probes: usize,
) -> NearestResult {
    let mut nearest_indexes = [0; NEAREST_COUNT];
    let mut nearest_distances = [u64::MAX; NEAREST_COUNT];
    let mut nearest_len = 0;
    let mut farthest_slot = 0;
    let current_worst_top_distance = Cell::new(u64::MAX);

    macro_rules! collect_candidates {
        ($method:ident) => {
            records.$method(
                vector,
                probes,
                &mut || current_worst_top_distance.get(),
                &mut |index, distance| {
                    if nearest_contains(&nearest_indexes, nearest_len, index) {
                        return;
                    }

                    if nearest_len < NEAREST_COUNT {
                        nearest_indexes[nearest_len] = index;
                        nearest_distances[nearest_len] = distance;
                        nearest_len += 1;

                        if distance > nearest_distances[farthest_slot] {
                            farthest_slot = nearest_len - 1;
                        }

                        if nearest_len == NEAREST_COUNT {
                            current_worst_top_distance.set(nearest_distances[farthest_slot]);
                        }
                    } else if distance < nearest_distances[farthest_slot] {
                        nearest_indexes[farthest_slot] = index;
                        nearest_distances[farthest_slot] = distance;
                        farthest_slot = farthest_slot_in(&nearest_distances);
                        current_worst_top_distance.set(nearest_distances[farthest_slot]);
                    }
                },
            );
        };
    }

    collect_candidates!(for_each_primary_candidate_probes);

    let boundary_case = is_ambiguous_nearest(&nearest_indexes, nearest_len, records);

    let mut nearest = Vec::with_capacity(nearest_len);

    for slot in 0..nearest_len {
        nearest.push((nearest_indexes[slot], nearest_distances[slot]));
    }

    nearest.sort_unstable_by_key(|(_, distance)| *distance);
    NearestResult {
        candidates: nearest,
        boundary_case,
    }
}

fn nearest_contains(
    nearest_indexes: &[usize; NEAREST_COUNT],
    nearest_len: usize,
    index: usize,
) -> bool {
    nearest_indexes[..nearest_len].contains(&index)
}

fn is_ambiguous_nearest(
    nearest_indexes: &[usize; NEAREST_COUNT],
    nearest_len: usize,
    records: &(impl ReferenceSource + ?Sized),
) -> bool {
    if nearest_len < NEAREST_COUNT {
        return true;
    }

    let frauds = nearest_indexes
        .iter()
        .filter(|index| records.is_fraud(**index))
        .count();

    frauds > 0 && frauds < NEAREST_COUNT
}

fn farthest_slot_in(distances: &[u64; NEAREST_COUNT]) -> usize {
    let mut farthest_slot = 0;

    for slot in 1..distances.len() {
        if distances[slot] > distances[farthest_slot] {
            farthest_slot = slot;
        }
    }

    farthest_slot
}

#[cfg(test)]
fn euclidean_distance2(a: &ReferenceVector, b: &ReferenceVector) -> u64 {
    a.iter()
        .zip(b)
        .map(|(left, right)| {
            let delta = i64::from(*left) - i64::from(*right);
            (delta * delta) as u64
        })
        .sum()
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

        fn for_each_primary_candidate_probes(
            &self,
            _vector: &ReferenceVector,
            probes: usize,
            _current_worst_top_distance: &mut dyn FnMut() -> u64,
            visit: &mut dyn FnMut(usize, u64),
        ) {
            let _ = probes;

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
            nearest(&query, &records, IVF_INITIAL_PROBES)
                .candidates
                .iter()
                .map(|(index, _)| *index)
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
                false, false, false, false, false, true, true, true, true, true,
            ],
            groups: [
                [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
                [(5, 0), (6, 1), (7, 2), (8, 3), (9, 4)],
            ],
        };

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
            nearest(&query, &records, IVF_INITIAL_PROBES)
                .candidates
                .iter()
                .map(|(index, _)| *index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }
}
