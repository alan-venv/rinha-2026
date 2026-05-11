pub struct RequestInput<'a> {
    pub transaction: Transaction,
    pub customer: Customer<'a>,
    pub merchant: Merchant<'a>,
    pub terminal: Terminal,
    pub last_transaction: Option<LastTransaction>,
}

#[derive(Clone, Copy)]
pub struct Transaction {
    pub amount: f64,
    pub installments: f64,
    pub requested_at: FastTimestamp,
}

pub struct Customer<'a> {
    pub avg_amount: f64,
    pub tx_count_24h: f64,
    pub known_merchants: &'a str,
}

pub struct Merchant<'a> {
    pub id: &'a str,
    pub mcc: &'a str,
    pub avg_amount: f64,
}

#[derive(Clone, Copy)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f64,
}

#[derive(Clone, Copy)]
pub struct LastTransaction {
    pub timestamp: FastTimestamp,
    pub km_from_current: f64,
}

#[derive(Clone, Copy)]
pub struct FastTimestamp {
    days_since_epoch: i32,
    pub hour: u8,
    seconds_since_midnight: u32,
    pub weekday_from_monday: u8,
}

impl FastTimestamp {
    pub fn parse(value: &str) -> Option<Self> {
        let bytes = value.as_bytes();

        let year = number(bytes, 0, 4)? as i32;
        let month = number(bytes, 5, 2)? as u8;
        let day = number(bytes, 8, 2)? as u8;
        let hour = number(bytes, 11, 2)? as u8;
        let minute = number(bytes, 14, 2)? as u8;
        let second = number(bytes, 17, 2)? as u8;

        let days_since_epoch = days_from_civil(year, month, day);
        let seconds_since_midnight =
            u32::from(hour) * 3_600 + u32::from(minute) * 60 + u32::from(second);
        let weekday_from_monday = (days_since_epoch + 3).rem_euclid(7) as u8;

        Some(Self {
            days_since_epoch,
            hour,
            seconds_since_midnight,
            weekday_from_monday,
        })
    }

    pub fn total_seconds(self) -> i64 {
        i64::from(self.days_since_epoch) * 86_400 + i64::from(self.seconds_since_midnight)
    }
}

fn number(bytes: &[u8], start: usize, len: usize) -> Option<u32> {
    let mut value = 0_u32;
    let digits = bytes.get(start..start + len)?;

    for byte in digits {
        value = value * 10 + u32::from(byte.wrapping_sub(b'0'));
    }

    Some(value)
}

fn days_from_civil(year: i32, month: u8, day: u8) -> i32 {
    let year = year - i32::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let shifted_month = i32::from(month) + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + i32::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;

    era * 146_097 + day_of_era - 719_468
}
