use chrono::{Datelike, Timelike};

use crate::consts::*;
use crate::dto::{ContentRequest, Customer, LastTransaction, Merchant, Transaction};
use crate::*;

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
}
