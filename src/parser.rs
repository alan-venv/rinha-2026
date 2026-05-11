use anyhow::{Result, anyhow};

use crate::dto::{
    Customer, FastTimestamp, LastTransaction, Merchant, RequestInput, Terminal, Transaction,
};

pub fn parse(bytes: &[u8]) -> Result<RequestInput<'_>> {
    JsonParser::new(bytes).parse_request()
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> JsonParser<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn parse_request(mut self) -> Result<RequestInput<'a>> {
        self.skip_whitespace();
        self.byte(b'{')?;
        self.key()?;
        self.string()?;
        self.comma()?;
        self.key()?;
        let transaction = self.parse_transaction()?;
        self.comma()?;
        self.key()?;
        let customer = self.parse_customer()?;
        self.comma()?;
        self.key()?;
        let merchant = self.parse_merchant()?;
        self.comma()?;
        self.key()?;
        let terminal = self.parse_terminal()?;
        self.comma()?;
        self.key()?;
        let last_transaction = self.parse_last_transaction()?;
        self.skip_whitespace();
        self.byte(b'}')?;
        self.skip_whitespace();
        if !self.is_finished() {
            return Err(anyhow!("trailing json data"));
        }

        Ok(RequestInput {
            transaction,
            customer,
            merchant,
            terminal,
            last_transaction,
        })
    }

    fn parse_transaction(&mut self) -> Result<Transaction> {
        self.skip_whitespace();
        self.byte(b'{')?;
        self.key()?;
        let amount = self.f64()?;
        self.comma()?;
        self.key()?;
        let installments = self.f64()?;
        self.comma()?;
        self.key()?;
        let requested_at =
            FastTimestamp::parse(self.string()?).ok_or_else(|| anyhow!("timestamp"))?;
        self.skip_whitespace();
        self.byte(b'}')?;

        Ok(Transaction {
            amount,
            installments,
            requested_at,
        })
    }

    fn parse_customer(&mut self) -> Result<Customer<'a>> {
        self.skip_whitespace();
        self.byte(b'{')?;
        self.key()?;
        let avg_amount = self.f64()?;
        self.comma()?;
        self.key()?;
        let tx_count_24h = self.f64()?;
        self.comma()?;
        self.key()?;
        let known_merchants = self.string_array()?;
        self.skip_whitespace();
        self.byte(b'}')?;

        Ok(Customer {
            avg_amount,
            tx_count_24h,
            known_merchants,
        })
    }

    fn parse_merchant(&mut self) -> Result<Merchant<'a>> {
        self.skip_whitespace();
        self.byte(b'{')?;
        self.key()?;
        let id = self.string()?;
        self.comma()?;
        self.key()?;
        let mcc = self.string()?;
        self.comma()?;
        self.key()?;
        let avg_amount = self.f64()?;
        self.skip_whitespace();
        self.byte(b'}')?;

        Ok(Merchant {
            id,
            mcc,
            avg_amount,
        })
    }

    fn parse_terminal(&mut self) -> Result<Terminal> {
        self.skip_whitespace();
        self.byte(b'{')?;
        self.key()?;
        let is_online = self.bool()?;
        self.comma()?;
        self.key()?;
        let card_present = self.bool()?;
        self.comma()?;
        self.key()?;
        let km_from_home = self.f64()?;
        self.skip_whitespace();
        self.byte(b'}')?;

        Ok(Terminal {
            is_online,
            card_present,
            km_from_home,
        })
    }

    fn parse_last_transaction(&mut self) -> Result<Option<LastTransaction>> {
        self.skip_whitespace();
        if self.literal(b"null") {
            return Ok(None);
        }

        self.byte(b'{')?;
        self.key()?;
        let timestamp = FastTimestamp::parse(self.string()?).ok_or_else(|| anyhow!("timestamp"))?;
        self.comma()?;
        self.key()?;
        let km_from_current = self.f64()?;
        self.skip_whitespace();
        self.byte(b'}')?;

        Ok(Some(LastTransaction {
            timestamp,
            km_from_current,
        }))
    }

    fn key(&mut self) -> Result<()> {
        self.string()?;
        self.skip_whitespace();
        self.byte(b':')
    }

    fn comma(&mut self) -> Result<()> {
        self.skip_whitespace();
        self.byte(b',')
    }

    fn string_array(&mut self) -> Result<&'a str> {
        self.skip_whitespace();
        self.byte(b'[')?;
        let start = self.position;
        let Some(relative_end) = self.bytes[start..].iter().position(|byte| *byte == b']') else {
            return Err(anyhow!("unterminated array"));
        };
        let end = start + relative_end;
        self.position = end + 1;

        std::str::from_utf8(&self.bytes[start..end]).map_err(|_| anyhow!("invalid utf-8 string"))
    }

    fn string(&mut self) -> Result<&'a str> {
        self.skip_whitespace();
        self.byte(b'"')?;
        let start = self.position;

        let Some(relative_end) = self.bytes[start..].iter().position(|byte| *byte == b'"') else {
            return Err(anyhow!("unterminated string"));
        };
        let end = start + relative_end;
        let value = &self.bytes[start..end];

        self.position = end + 1;
        std::str::from_utf8(value).map_err(|_| anyhow!("invalid utf-8 string"))
    }

    fn f64(&mut self) -> Result<f64> {
        self.skip_whitespace();
        let start = self.position;
        self.skip_number()?;
        let value = std::str::from_utf8(&self.bytes[start..self.position])
            .map_err(|_| anyhow!("invalid number"))?;

        value.parse().map_err(|_| anyhow!("invalid f64"))
    }

    fn bool(&mut self) -> Result<bool> {
        self.skip_whitespace();
        if self.literal(b"true") {
            return Ok(true);
        }

        if self.literal(b"false") {
            return Ok(false);
        }

        Err(anyhow!("invalid bool"))
    }

    fn skip_number(&mut self) -> Result<()> {
        let start = self.position;

        while !matches!(
            self.peek(),
            None | Some(b',' | b'}' | b']' | b' ' | b'\n' | b'\r' | b'\t')
        ) {
            self.position += 1;
        }

        if self.position == start {
            return Err(anyhow!("empty number"));
        }

        Ok(())
    }

    fn literal(&mut self, value: &[u8]) -> bool {
        if self.bytes[self.position..].starts_with(value) {
            self.position += value.len();
            true
        } else {
            false
        }
    }

    fn byte(&mut self, expected: u8) -> Result<()> {
        match self.next() {
            Some(byte) if byte == expected => Ok(()),
            _ => Err(anyhow!("unexpected json byte")),
        }
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.position += 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.position == self.bytes.len()
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.position += 1;
        Some(byte)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn parses_request() {
        let request = parse(sample_json()).unwrap();

        assert_eq!(request.transaction.amount, 384.88);
        assert!(request.customer.known_merchants.contains("MERC-001"));
        assert_eq!(request.merchant.id, "MERC-001");
        assert!(request.last_transaction.is_some());
    }

    #[test]
    fn parses_null_last_transaction() {
        let json = br#"{
            "id": "tx-1",
            "transaction": {
                "amount": 384.88,
                "installments": 3,
                "requested_at": "2026-03-11T20:23:35Z"
            },
            "customer": {
                "avg_amount": 769.76,
                "tx_count_24h": 3,
                "known_merchants": ["MERC-001"]
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

        let request = parse(json).unwrap();
        assert!(request.last_transaction.is_none());
    }

    #[test]
    fn parses_fast_timestamp_components() {
        let timestamp = FastTimestamp::parse("2026-03-11T20:23:35Z").unwrap();

        assert_eq!(timestamp.hour, 20);
        assert_eq!(timestamp.weekday_from_monday, 2);
        assert_eq!(timestamp.total_seconds(), 1_773_260_615);
    }

    #[test]
    fn returns_error_for_structurally_invalid_json() {
        assert!(parse(br#"{"transaction":"#).is_err());
    }
}
