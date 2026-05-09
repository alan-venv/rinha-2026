use std::fs::File;
use std::time::Instant;

use anyhow::Result;
use rinha::dto::ContentRequest;
use rinha::memory;
use rinha::{encoding, service};
use serde::{Deserialize, Serialize};

const MAX_ELAPSED_MS: u128 = 6_000;
const MAX_BOUNDARY_CASE_PERCENTAGE: f64 = 10.0;

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
struct DiagnoseOutput {
    core: DiagnoseCore,
    warnings: DiagnoseWarnings,
}

#[derive(Serialize)]
struct DiagnoseCore {
    entries: usize,
    elapsed_ms: u128,
    boundary_cases: usize,
    boundary_case_percentage: String,
    decision_mismatches: usize,
    decision_mismatch_percentage: String,
    boundary_decision_mismatches: usize,
    primary_only_decision_mismatches: usize,
}

#[derive(Serialize)]
struct DiagnoseWarnings {
    items: Vec<DiagnoseWarning>,
}

#[derive(Serialize)]
struct DiagnoseWarning {
    metric: &'static str,
    actual: String,
    limit: String,
}

fn main() -> Result<()> {
    let references = memory::load_references()?;
    let input = File::open("scripts/k6/data.json")?;
    let data: TestData = serde_json::from_reader(input)?;
    let total = data.entries.len();
    let started = Instant::now();
    let mut decision_mismatches = 0;
    let mut boundary_cases = 0;
    let mut boundary_decision_mismatches = 0;
    let mut primary_only_decision_mismatches = 0;

    for entry in data.entries {
        let vector = encoding::vectorization(entry.request);
        let details = service::fraud_score_details(&vector, &references);
        let approved = details.score < 0.6;

        if details.boundary_case {
            boundary_cases += 1;
        }

        if approved == entry.expected_approved {
            continue;
        }

        decision_mismatches += 1;
        if details.boundary_case {
            boundary_decision_mismatches += 1;
        } else {
            primary_only_decision_mismatches += 1;
        }
    }

    let core = DiagnoseCore {
        entries: total,
        elapsed_ms: started.elapsed().as_millis(),
        boundary_cases,
        boundary_case_percentage: percentage(boundary_cases, total),
        decision_mismatches,
        decision_mismatch_percentage: percentage(decision_mismatches, total),
        boundary_decision_mismatches,
        primary_only_decision_mismatches,
    };
    let output = DiagnoseOutput {
        warnings: diagnose_warnings(&core),
        core,
    };

    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}

fn diagnose_warnings(core: &DiagnoseCore) -> DiagnoseWarnings {
    let mut items = Vec::new();

    if core.elapsed_ms > MAX_ELAPSED_MS {
        items.push(DiagnoseWarning {
            metric: "elapsed_ms",
            actual: core.elapsed_ms.to_string(),
            limit: MAX_ELAPSED_MS.to_string(),
        });
    }

    let boundary_case_percentage = percentage_value(core.boundary_cases, core.entries);
    if boundary_case_percentage > MAX_BOUNDARY_CASE_PERCENTAGE {
        items.push(DiagnoseWarning {
            metric: "boundary_case_percentage",
            actual: format!("{boundary_case_percentage:.2}%"),
            limit: format!("{MAX_BOUNDARY_CASE_PERCENTAGE:.2}%"),
        });
    }

    if core.primary_only_decision_mismatches > 0 {
        items.push(DiagnoseWarning {
            metric: "primary_only_decision_mismatches",
            actual: core.primary_only_decision_mismatches.to_string(),
            limit: "0".to_string(),
        });
    }

    DiagnoseWarnings { items }
}

fn percentage(value: usize, total: usize) -> String {
    format!("{:.2}%", percentage_value(value, total))
}

fn percentage_value(value: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }

    ((value as f64 / total as f64) * 10_000.0).round() / 100.0
}
