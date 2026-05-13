use std::fs::File;
use std::time::Instant;

use anyhow::Result;
use rinha::{
    encoding,
    kdtree::KdTree,
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
    morton_cases: usize,
    kdtree_cases: usize,
    morton_mismatches: usize,
    kdtree_mismatches: usize,
}

fn main() -> Result<()> {
    let index = MortonIndex::load_default()?;
    let kdtree = KdTree::load_default()?;
    let service = Service::new(index, kdtree);
    let input = File::open("scripts/k6/data.json")?;
    let data: TestData = serde_json::from_reader(input)?;
    let total = data.entries.len();

    let started = Instant::now();
    let mut morton_cases = 0;
    let mut kdtree_cases = 0;
    let mut morton_mismatches = 0;
    let mut kdtree_mismatches = 0;

    for entry in data.entries {
        let request = parser::parse(entry.request.get().as_bytes())?;
        let vector = encoding::vectorization(&request);
        let decision = service.decide(&vector);
        let approved = decision.fraud_score < 0.6;
        let mismatched = approved != entry.expected_approved;

        match decision.source {
            DecisionSource::Morton => {
                morton_cases += 1;
                morton_mismatches += usize::from(mismatched);
            }
            DecisionSource::KdTree => {
                kdtree_cases += 1;
                kdtree_mismatches += usize::from(mismatched);
            }
        }
    }

    let diagnose = Diagnose {
        entries: total,
        elapsed_ms: started.elapsed().as_millis(),
        morton_cases,
        kdtree_cases,
        morton_mismatches,
        kdtree_mismatches,
    };

    println!("{}", serde_json::to_string_pretty(&diagnose)?);

    Ok(())
}
