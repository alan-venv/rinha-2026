use chrono::{Datelike, Timelike};

use crate::dto::{ContentRequest, LastTransaction, Transaction};
use crate::*;

const N: [f64; 7] = [10000.0, 12.0, 10.0, 1440.0, 1000.0, 20.0, 10000.0];

const R0: f32 = 0.5;
const R: &[(u16, f32)] = &[
    (4511, 0.35),
    (5311, 0.25),
    (5411, 0.15),
    (5812, 0.30),
    (5912, 0.20),
    (5944, 0.45),
    (5999, 0.50),
    (7801, 0.80),
    (7802, 0.75),
    (7995, 0.85),
];

pub fn vectorization(request: ContentRequest) -> ReferenceVector {
    let transaction = &request.transaction;
    let last_transaction = &request.last_transaction;
    let customer = &request.customer;
    let merchant = &request.merchant;
    let terminal = &request.terminal;

    let n0 = c(transaction.amount / N[0]);
    let n1 = c(transaction.installments / N[1]);
    let n2 = c((transaction.amount / customer.avg_amount) / N[2]);
    let n3 = transaction.requested_at.hour() as f32 / 23.0;
    let n4 = transaction.requested_at.weekday().num_days_from_monday() as f32 / 6.0;
    let n5 = f5(transaction, last_transaction);
    let n6 = f6(last_transaction);
    let n7 = c(terminal.km_from_home / N[4]);
    let n8 = c(customer.tx_count_24h / N[5]);
    let n9: f32 = if terminal.is_online { 1.0 } else { 0.0 };
    let n10: f32 = if terminal.card_present { 1.0 } else { 0.0 };
    let n11 = f11(&customer.known_merchants, &merchant.id);
    let n12 = f12(&merchant.mcc);
    let n13 = c(merchant.avg_amount / N[6]);

    [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13].map(q)
}

fn c(x: f64) -> f32 {
    x.clamp(0.0, 1.0) as f32
}

fn q(x: f32) -> i16 {
    (x * VECTOR_SCALE).round() as i16
}

fn f5(t: &Transaction, l: &Option<LastTransaction>) -> f32 {
    l.as_ref().map_or(-1.0, |x| {
        c(t.requested_at
            .signed_duration_since(x.timestamp)
            .num_minutes()
            .max(0) as f64
            / N[3])
    })
}

fn f6(x: &Option<LastTransaction>) -> f32 {
    x.as_ref().map_or(-1.0, |y| c(y.km_from_current / N[4]))
}

fn f11(xs: &[String], x: &str) -> f32 {
    if xs.iter().any(|y| y == x) { 0.0 } else { 1.0 }
}

fn f12(x: &str) -> f32 {
    let Ok(x) = x.parse::<u16>() else {
        return R0;
    };

    match R.binary_search_by_key(&x, |(y, _)| *y) {
        Ok(i) => R[i].1,
        Err(_) => R0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dto::{Customer, Merchant};
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
        assert_eq!(c(-0.1), 0.0);
        assert_eq!(c(-10.0), 0.0);
    }

    #[test]
    fn clamps_values_above_one() {
        assert_eq!(c(1.1), 1.0);
        assert_eq!(c(10.0), 1.0);
    }

    #[test]
    fn keeps_values_between_zero_and_one() {
        assert_eq!(c(0.0), 0.0);
        assert_eq!(c(0.5), 0.5);
        assert_eq!(c(1.0), 1.0);
    }

    #[test]
    fn quantizes_to_four_decimal_places() {
        assert_eq!(q(0.038488), 385);
        assert_eq!(q(0.33333334), 3333);
        assert_eq!(q(1.0), 10_000);
        assert_eq!(q(-1.0), -10_000);
    }

    #[test]
    fn calculates_normalized_minutes_since_last_transaction() {
        let transaction = transaction(utc(2026, 3, 11, 20, 23, 35));
        let last_transaction = LastTransaction {
            timestamp: utc(2026, 3, 11, 14, 58, 35),
            km_from_current: 0.0,
        };

        assert_eq!(
            f5(&transaction, &Some(last_transaction)),
            (325.0 / N[3]) as f32
        );
    }

    #[test]
    fn returns_minus_one_when_last_transaction_is_missing() {
        let transaction = transaction(utc(2026, 3, 11, 20, 23, 35));

        assert_eq!(f5(&transaction, &None), -1.0);
    }

    #[test]
    fn does_not_return_negative_minutes() {
        let transaction = transaction(utc(2026, 3, 11, 20, 23, 35));
        let last_transaction = LastTransaction {
            timestamp: utc(2026, 3, 11, 21, 23, 35),
            km_from_current: 0.0,
        };

        assert_eq!(f5(&transaction, &Some(last_transaction)), 0.0);
    }

    #[test]
    fn calculates_normalized_km_from_last_transaction() {
        let last_transaction = LastTransaction {
            timestamp: utc(2026, 3, 11, 14, 58, 35),
            km_from_current: 250.0,
        };

        assert_eq!(f6(&Some(last_transaction)), 0.25);
    }

    #[test]
    fn clamps_km_from_last_transaction() {
        let last_transaction = LastTransaction {
            timestamp: utc(2026, 3, 11, 14, 58, 35),
            km_from_current: 2_000.0,
        };

        assert_eq!(f6(&Some(last_transaction)), 1.0);
    }

    #[test]
    fn returns_minus_one_when_last_transaction_km_is_missing() {
        assert_eq!(f6(&None), -1.0);
    }

    #[test]
    fn returns_zero_for_known_merchant() {
        let customer = customer(vec!["MERC-001".to_string()]);
        let merchant = merchant("MERC-001");

        assert_eq!(f11(&customer.known_merchants, &merchant.id), 0.0);
    }

    #[test]
    fn returns_one_for_unknown_merchant() {
        let customer = customer(vec!["MERC-001".to_string()]);
        let merchant = merchant("MERC-002");

        assert_eq!(f11(&customer.known_merchants, &merchant.id), 1.0);
    }

    #[test]
    fn mcc_risk_is_sorted_for_binary_search() {
        assert!(R.windows(2).all(|pair| pair[0].0 < pair[1].0));
    }

    #[test]
    fn finds_known_mcc_risk() {
        assert_eq!(f12("7995"), 0.85);
        assert_eq!(f12("5411"), 0.15);
    }

    #[test]
    fn uses_default_for_unknown_or_invalid_mcc() {
        assert_eq!(f12("0000"), R0);
        assert_eq!(f12("invalid"), R0);
    }
}
