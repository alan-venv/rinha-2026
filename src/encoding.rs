use crate::dto::{LastTransaction, RequestInput};

const N: [f64; 7] = [10000.0, 12.0, 10.0, 1440.0, 1000.0, 20.0, 10000.0];
const R0: f32 = 0.5;
const RK: [u16; 10] = [4511, 5311, 5411, 5812, 5912, 5944, 5999, 7801, 7802, 7995];
const RV: [f32; 10] = [0.35, 0.25, 0.15, 0.30, 0.20, 0.45, 0.50, 0.80, 0.75, 0.85];

pub fn vectorization(request: &RequestInput<'_>) -> [i16; 16] {
    let transaction = &request.transaction;
    let last_transaction = request.last_transaction;
    let customer = &request.customer;
    let merchant = &request.merchant;
    let terminal = &request.terminal;

    let n0 = n0(transaction.amount);
    let n1 = n1(transaction.installments);
    let n2 = n2(transaction.amount, customer.avg_amount);
    let n3 = n3_input(transaction.requested_at.hour);
    let n4 = n4_input(transaction.requested_at.weekday_from_monday);
    let n5 = n5_input(transaction.requested_at.total_seconds(), last_transaction);
    let n6 = n6_input(last_transaction);
    let n7 = n7(terminal.km_from_home);
    let n8 = n8(customer.tx_count_24h);
    let n9 = n9(terminal.is_online);
    let n10 = n10(terminal.card_present);
    let n11 = n11_input(customer.known_merchants.contains(merchant.id));
    let n12 = n12(merchant.mcc);
    let n13 = n13(merchant.avg_amount);

    [
        n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, 0.0, 0.0,
    ]
    .map(q)
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

fn n3_input(hour: u8) -> f32 {
    hour as f32 / 23.0
}

fn n4_input(weekday_from_monday: u8) -> f32 {
    weekday_from_monday as f32 / 6.0
}

fn n5_input(requested_at_seconds: i64, last_transaction: Option<LastTransaction>) -> f32 {
    last_transaction.map_or(-1.0, |last_transaction| {
        let minutes =
            ((requested_at_seconds - last_transaction.timestamp.total_seconds()) / 60).max(0);

        c(minutes as f64 / N[3])
    })
}

fn n6_input(last_transaction: Option<LastTransaction>) -> f32 {
    last_transaction.map_or(
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

fn n11_input(merchant_known: bool) -> f32 {
    (!merchant_known) as u8 as f32
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
    use crate::dto::FastTimestamp;

    fn sample_json() -> &'static [u8] {
        br#"{
            "id": "tx-3576980410",
            "transaction": {
                "amount": 384.88,
                "installments": 3,
                "requested_at": "2026-03-11T20:23:35Z"
            },
            "customer": {
                "avg_amount": 769.76,
                "tx_count_24h": 3,
                "known_merchants": ["MERC-009", "MERC-009", "MERC-001", "MERC-001"]
            },
            "merchant": {
                "id": "MERC-001",
                "mcc": "5912",
                "avg_amount": 298.95
            },
            "terminal": {
                "is_online": false,
                "card_present": true,
                "km_from_home": 13.7090520965
            },
            "last_transaction": {
                "timestamp": "2026-03-11T14:58:35Z",
                "km_from_current": 18.8626479774
            }
        }"#
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
        let requested_at = FastTimestamp::parse("2026-03-11T20:23:35Z").unwrap();
        let last_transaction = LastTransaction {
            timestamp: FastTimestamp::parse("2026-03-11T14:58:35Z").unwrap(),
            km_from_current: 0.0,
        };

        assert_eq!(
            n5_input(requested_at.total_seconds(), Some(last_transaction)),
            (325.0 / N[3]) as f32
        );
    }

    #[test]
    fn returns_minus_one_when_last_transaction_is_missing() {
        let requested_at = FastTimestamp::parse("2026-03-11T20:23:35Z").unwrap();

        assert_eq!(n5_input(requested_at.total_seconds(), None), -1.0);
    }

    #[test]
    fn does_not_return_negative_minutes() {
        let requested_at = FastTimestamp::parse("2026-03-11T20:23:35Z").unwrap();
        let last_transaction = LastTransaction {
            timestamp: FastTimestamp::parse("2026-03-11T21:23:35Z").unwrap(),
            km_from_current: 0.0,
        };

        assert_eq!(
            n5_input(requested_at.total_seconds(), Some(last_transaction)),
            0.0
        );
    }

    #[test]
    fn calculates_normalized_km_from_last_transaction() {
        let last_transaction = LastTransaction {
            timestamp: FastTimestamp::parse("2026-03-11T14:58:35Z").unwrap(),
            km_from_current: 250.0,
        };

        assert_eq!(n6_input(Some(last_transaction)), 0.25);
    }

    #[test]
    fn clamps_km_from_last_transaction() {
        let last_transaction = LastTransaction {
            timestamp: FastTimestamp::parse("2026-03-11T14:58:35Z").unwrap(),
            km_from_current: 2_000.0,
        };

        assert_eq!(n6_input(Some(last_transaction)), 1.0);
    }

    #[test]
    fn returns_minus_one_when_last_transaction_km_is_missing() {
        assert_eq!(n6_input(None), -1.0);
    }

    #[test]
    fn returns_zero_for_known_merchant() {
        let request = crate::parser::parse(sample_json()).unwrap();

        assert_eq!(vectorization(&request)[11], 0);
    }

    #[test]
    fn returns_one_for_unknown_merchant() {
        let unknown = br#"{
            "id": "tx-1",
            "transaction": {
                "amount": 384.88,
                "installments": 3,
                "requested_at": "2026-03-11T20:23:35Z"
            },
            "customer": {
                "avg_amount": 769.76,
                "tx_count_24h": 3,
                "known_merchants": ["MERC-009"]
            },
            "merchant": {
                "id": "MERC-001",
                "mcc": "5912",
                "avg_amount": 298.95
            },
            "terminal": {
                "is_online": false,
                "card_present": true,
                "km_from_home": 13.7
            },
            "last_transaction": null
        }"#;
        let request = crate::parser::parse(unknown).unwrap();

        assert_eq!(vectorization(&request)[11], 10_000);
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
    fn vector_from_request_matches_expected_sample_dimensions() {
        let request = crate::parser::parse(sample_json()).unwrap();
        let vector = vectorization(&request);

        assert_eq!(vector[0], 385);
        assert_eq!(vector[1], 2500);
        assert_eq!(vector[2], 500);
        assert_eq!(vector[3], 8696);
        assert_eq!(vector[4], 3333);
        assert_eq!(vector[5], 2257);
        assert_eq!(vector[6], 189);
        assert_eq!(vector[7], 137);
        assert_eq!(vector[8], 1500);
        assert_eq!(vector[9], 0);
        assert_eq!(vector[10], 10_000);
        assert_eq!(vector[11], 0);
        assert_eq!(vector[12], 2000);
        assert_eq!(vector[13], 299);
        assert_eq!(vector[14], 0);
        assert_eq!(vector[15], 0);
    }
}
