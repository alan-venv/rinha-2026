pub fn fraud_response(fraud_score: f32) -> &'static [u8] {
    const RESPONSES: [&[u8]; 6] = [
        br#"{"approved":true,"fraud_score":0.0}"#,
        br#"{"approved":true,"fraud_score":0.2}"#,
        br#"{"approved":true,"fraud_score":0.4}"#,
        br#"{"approved":false,"fraud_score":0.6}"#,
        br#"{"approved":false,"fraud_score":0.8}"#,
        br#"{"approved":false,"fraud_score":1.0}"#,
    ];

    RESPONSES[(fraud_score * 5.0).round() as usize]
}
