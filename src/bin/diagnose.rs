use std::fs::File;
use std::time::Instant;

use anyhow::Result;
use rinha::dto::ContentRequest;
use rinha::{encoding, service};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct TestData {
    entries: Vec<TestEntry>,
}

#[derive(Deserialize)]
struct TestEntry {
    request: ContentRequest,
    expected_approved: bool,
}

#[derive(Serialize)]
struct Diagnose {
    entries: usize,
    elapsed_ms: u128,
    decision_mismatches: usize,
}

fn main() -> Result<()> {
    //let references = ... ; // carrega as referencias que foram buildadas pelo /src/bin/index
    let input = File::open("scripts/k6/data.json")?;
    let data: TestData = serde_json::from_reader(input)?;
    let total = data.entries.len();

    let started = Instant::now();
    let mut decision_mismatches = 0;

    for entry in data.entries {
        let vector = encoding::vectorization(entry.request);
        let score = service::fraud_score(&vector); // ,&references // injeta as referencias necessárias para o cálculo
        let approved = score < 0.6;

        if approved == entry.expected_approved {
            continue;
        }

        decision_mismatches += 1;
    }

    let diagnose = Diagnose {
        entries: total,
        elapsed_ms: started.elapsed().as_millis(),
        decision_mismatches,
    };

    println!("{}", serde_json::to_string_pretty(&diagnose)?);

    Ok(())
}
