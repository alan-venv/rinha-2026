pub const MAX_AMOUNT: f64 = 10000.0;
pub const MAX_INSTALLMENTS: f64 = 12.0;
pub const AMOUNT_VS_AVG_RATIO: f64 = 10.0;
pub const MAX_MINUTES: f64 = 1440.0;
pub const MAX_KM: f64 = 1000.0;
pub const MAX_TX_COUNT_24H: f64 = 20.0;
pub const MAX_MERCHANT_AVG_AMOUNT: f64 = 10000.0;

const DEFAULT_MCC_RISK: f32 = 0.5;
const MCC_RISK: &[(u16, f32)] = &[
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

#[inline]
pub fn mcc_risk(mcc: &str) -> f32 {
    let Ok(code) = mcc.parse::<u16>() else {
        return DEFAULT_MCC_RISK;
    };

    match MCC_RISK.binary_search_by_key(&code, |(mcc, _)| *mcc) {
        Ok(index) => MCC_RISK[index].1,
        Err(_) => DEFAULT_MCC_RISK,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcc_risk_is_sorted_for_binary_search() {
        assert!(MCC_RISK.windows(2).all(|pair| pair[0].0 < pair[1].0));
    }

    #[test]
    fn finds_known_mcc_risk() {
        assert_eq!(mcc_risk("7995"), 0.85);
        assert_eq!(mcc_risk("5411"), 0.15);
    }

    #[test]
    fn uses_default_for_unknown_or_invalid_mcc() {
        assert_eq!(mcc_risk("0000"), DEFAULT_MCC_RISK);
        assert_eq!(mcc_risk("invalid"), DEFAULT_MCC_RISK);
    }
}
