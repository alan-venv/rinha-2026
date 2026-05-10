use chrono::{DateTime, Datelike, Timelike, Utc};

use crate::dto::{ContentRequest, LastTransaction};
use crate::*;

const N: [f64; 7] = [10000.0, 12.0, 10.0, 1440.0, 1000.0, 20.0, 10000.0];
const R0: f32 = 0.5;
const RK: [u16; 10] = [4511, 5311, 5411, 5812, 5912, 5944, 5999, 7801, 7802, 7995];
const RV: [f32; 10] = [0.35, 0.25, 0.15, 0.30, 0.20, 0.45, 0.50, 0.80, 0.75, 0.85];

pub fn vectorization(request: ContentRequest) -> ReferenceVector {
    let transaction = &request.transaction;
    let last_transaction = &request.last_transaction;
    let customer = &request.customer;
    let merchant = &request.merchant;
    let terminal = &request.terminal;

    let n0 = n0(transaction.amount);
    let n1 = n1(transaction.installments);
    let n2 = n2(transaction.amount, customer.avg_amount);
    let n3 = n3(&transaction.requested_at);
    let n4 = n4(&transaction.requested_at);
    let n5 = n5(&transaction.requested_at, last_transaction);
    let n6 = n6(last_transaction);
    let n7 = n7(terminal.km_from_home);
    let n8 = n8(customer.tx_count_24h);
    let n9 = n9(terminal.is_online);
    let n10 = n10(terminal.card_present);
    let n11 = n11(&customer.known_merchants, &merchant.id);
    let n12 = n12(&merchant.mcc);
    let n13 = n13(merchant.avg_amount);

    [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13].map(q)
}

fn n0(amount: f64) -> f32 {
    c(amount / N[0])
}

fn n1(installments: f64) -> f32 {
    c(installments / N[1])
}

fn n2(amount: f64, avg_amount: f64) -> f32 {
    c((amount / avg_amount) / N[2])
}

fn n3(requested_at: &DateTime<Utc>) -> f32 {
    requested_at.hour() as f32 / 23.0
}

fn n4(requested_at: &DateTime<Utc>) -> f32 {
    requested_at.weekday().num_days_from_monday() as f32 / 6.0
}

fn n5(requested_at: &DateTime<Utc>, last_transaction: &Option<LastTransaction>) -> f32 {
    last_transaction.as_ref().map_or(-1.0, |last_transaction| {
        c(requested_at
            .signed_duration_since(last_transaction.timestamp)
            .num_minutes()
            .max(0) as f64
            / N[3])
    })
}

fn n6(last_transaction: &Option<LastTransaction>) -> f32 {
    last_transaction.as_ref().map_or(
        -1.0,
        |last_transaction| n7(last_transaction.km_from_current),
    )
}

fn n7(km_from_home: f64) -> f32 {
    c(km_from_home / N[4])
}

fn n8(tx_count_24h: f64) -> f32 {
    c(tx_count_24h / N[5])
}

fn n9(is_online: bool) -> f32 {
    is_online as u8 as f32
}

fn n10(card_present: bool) -> f32 {
    card_present as u8 as f32
}

fn n11(known_merchants: &[String], merchant_id: &str) -> f32 {
    (!known_merchants.iter().any(|known| known == merchant_id)) as u8 as f32
}

fn n12(mcc: &str) -> f32 {
    let Ok(mcc) = mcc.parse::<u16>() else {
        return R0;
    };

    match RK.binary_search(&mcc) {
        Ok(index) => RV[index],
        Err(_) => R0,
    }
}

fn n13(avg_amount: f64) -> f32 {
    c(avg_amount / N[6])
}

fn c(x: f64) -> f32 {
    x.clamp(0.0, 1.0) as f32
}

fn q(x: f32) -> i16 {
    (x * 10_000.0).round() as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dto::{Customer, Merchant, Transaction};
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
            n5(&transaction.requested_at, &Some(last_transaction)),
            (325.0 / N[3]) as f32
        );
    }

    #[test]
    fn returns_minus_one_when_last_transaction_is_missing() {
        let transaction = transaction(utc(2026, 3, 11, 20, 23, 35));

        assert_eq!(n5(&transaction.requested_at, &None), -1.0);
    }

    #[test]
    fn does_not_return_negative_minutes() {
        let transaction = transaction(utc(2026, 3, 11, 20, 23, 35));
        let last_transaction = LastTransaction {
            timestamp: utc(2026, 3, 11, 21, 23, 35),
            km_from_current: 0.0,
        };

        assert_eq!(n5(&transaction.requested_at, &Some(last_transaction)), 0.0);
    }

    #[test]
    fn calculates_normalized_km_from_last_transaction() {
        let last_transaction = LastTransaction {
            timestamp: utc(2026, 3, 11, 14, 58, 35),
            km_from_current: 250.0,
        };

        assert_eq!(n6(&Some(last_transaction)), 0.25);
    }

    #[test]
    fn clamps_km_from_last_transaction() {
        let last_transaction = LastTransaction {
            timestamp: utc(2026, 3, 11, 14, 58, 35),
            km_from_current: 2_000.0,
        };

        assert_eq!(n6(&Some(last_transaction)), 1.0);
    }

    #[test]
    fn returns_minus_one_when_last_transaction_km_is_missing() {
        assert_eq!(n6(&None), -1.0);
    }

    #[test]
    fn returns_zero_for_known_merchant() {
        let customer = customer(vec!["MERC-001".to_string()]);
        let merchant = merchant("MERC-001");

        assert_eq!(n11(&customer.known_merchants, &merchant.id), 0.0);
    }

    #[test]
    fn returns_one_for_unknown_merchant() {
        let customer = customer(vec!["MERC-001".to_string()]);
        let merchant = merchant("MERC-002");

        assert_eq!(n11(&customer.known_merchants, &merchant.id), 1.0);
    }

    #[test]
    fn mcc_risk_is_sorted_for_binary_search() {
        assert!(RK.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn finds_known_mcc_risk() {
        assert_eq!(n12("7995"), 0.85);
        assert_eq!(n12("5411"), 0.15);
    }

    #[test]
    fn uses_default_for_unknown_or_invalid_mcc() {
        assert_eq!(n12("0000"), R0);
        assert_eq!(n12("invalid"), R0);
    }

    #[test]
    fn vectorization_returns_fourteen_dimensions() {
        let request = ContentRequest {
            id: "tx".to_string(),
            transaction: transaction(utc(2026, 3, 11, 20, 23, 35)),
            customer: customer(vec!["MERC-001".to_string()]),
            merchant: merchant("MERC-001"),
            terminal: crate::dto::Terminal {
                is_online: true,
                card_present: false,
                km_from_home: 10.0,
            },
            last_transaction: None,
        };

        let vector = vectorization(request);

        assert_eq!(vector.len(), 14);
    }
}
