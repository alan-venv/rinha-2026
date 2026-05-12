use std::fs::File;
use std::time::Instant;

use anyhow::Result;
use rinha::{
    encoding,
    morton::MortonIndex,
    parser,
    service::{DecisionSource, Service},
};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

#[derive(Deserialize)]
struct TestData {
    entries: Vec<TestEntry>,
}

#[derive(Deserialize)]
struct TestEntry {
    request: Box<RawValue>,
    expected_approved: bool,
}

#[derive(Serialize)]
struct Diagnose {
    entries: usize,
    elapsed_ms: u128,
    decision_mismatches: usize,
    boundary_cases: usize,
    morton_mismatches: usize,
}

fn main() -> Result<()> {
    let index = MortonIndex::load_default()?;
    let service = Service::new(index);
    let input = File::open("scripts/k6/data.json")?;
    let data: TestData = serde_json::from_reader(input)?;
    let total = data.entries.len();

    let started = Instant::now();
    let mut decision_mismatches = 0;
    let mut boundary_cases = 0;
    let mut morton_mismatches = 0;

    for entry in data.entries {
        let request = parser::parse(entry.request.get().as_bytes())?;
        let vector = encoding::vectorization(&request);
        let decision = service.decide(&vector);
        let approved = decision.fraud_score < 0.6;
        let mismatched = approved != entry.expected_approved;
        decision_mismatches += usize::from(mismatched);

        match decision.source {
            DecisionSource::Morton => {
                morton_mismatches += usize::from(mismatched);
            }
            DecisionSource::Boundary => {
                boundary_cases += 1;
            }
        }
    }

    let diagnose = Diagnose {
        entries: total,
        elapsed_ms: started.elapsed().as_millis(),
        decision_mismatches,
        boundary_cases,
        morton_mismatches,
    };

    println!("{}", serde_json::to_string_pretty(&diagnose)?);

    Ok(())
}
