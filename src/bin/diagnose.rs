use std::fs::File;
use std::time::Instant;

use anyhow::Result;
use rinha::dto::ContentRequest;
use rinha::memory::{self, ReferenceSource};
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
    candidates: DiagnoseCandidates,
    centroids: DiagnoseCentroids,
}

#[derive(Serialize)]
struct DiagnoseCore {
    entries: usize,
    elapsed_ms: u128,
    hierarchy_build_elapsed_ms: u128,
    boundary_cases: usize,
    boundary_case_percentage: String,
    decision_mismatches: usize,
    decision_mismatch_percentage: String,
    boundary_decision_mismatches: usize,
    primary_only_decision_mismatches: usize,
    avg_search_cost_units: usize,
    max_search_cost_units: usize,
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

#[derive(Default)]
struct MetricAccumulator {
    sum: usize,
    count: usize,
    max: usize,
}

impl MetricAccumulator {
    fn new() -> Self {
        Self::default()
    }

    fn record(&mut self, value: usize) {
        self.sum += value;
        self.count += 1;
        self.max = self.max.max(value);
    }

    fn average(&self) -> usize {
        if self.count == 0 {
            return 0;
        }

        (self.sum as f64 / self.count as f64).round() as usize
    }

    fn max(&self) -> usize {
        self.max
    }
}

#[derive(Serialize)]
struct DiagnoseCandidates {
    avg_primary_list_candidates: usize,
    max_primary_list_candidates: usize,
    avg_flat_candidates: usize,
    avg_flat_early_discards: usize,
    avg_flat_full_distance_candidates: usize,
    avg_flat_vector_dimensions_evaluated: usize,
}

#[derive(Serialize)]
struct DiagnoseCentroids {
    avg_coarse_centroid_candidates: usize,
    avg_fine_centroid_candidates: usize,
    avg_centroid_candidates: usize,
    avg_centroid_early_discards: usize,
    avg_centroid_full_distance_candidates: usize,
    avg_centroid_vector_dimensions_evaluated: usize,
}

fn main() -> Result<()> {
    let started = Instant::now();
    let references = memory::load_references()?;
    let input = File::open("scripts/k6/data.json")?;
    let data: TestData = serde_json::from_reader(input)?;
    let total = data.entries.len();
    let mut decision_mismatches = 0;
    let mut boundary_cases = 0;
    let mut boundary_decision_mismatches = 0;
    let mut primary_only_decision_mismatches = 0;
    let mut primary_list_candidates = MetricAccumulator::new();
    let mut coarse_centroid_candidates = MetricAccumulator::new();
    let mut fine_centroid_candidates = MetricAccumulator::new();
    let mut centroid_candidates = MetricAccumulator::new();
    let mut centroid_early_discards = MetricAccumulator::new();
    let mut centroid_full_distance_candidates = MetricAccumulator::new();
    let mut centroid_vector_dimensions_evaluated = MetricAccumulator::new();
    let mut flat_candidates = MetricAccumulator::new();
    let mut flat_early_discards = MetricAccumulator::new();
    let mut flat_full_distance_candidates = MetricAccumulator::new();
    let mut flat_vector_dimensions_evaluated = MetricAccumulator::new();
    let mut search_cost_units = MetricAccumulator::new();

    for entry in data.entries {
        let vector = encoding::vectorization(entry.request);
        let context = references.prepare_search_context(&vector);
        let details = service::fraud_score_details(&vector, &references);
        let primary = references.search_cost_for_probe_count_with_context(
            &context,
            &vector,
            rinha::IVF_FINE_PROBES,
        );
        let cost = primary.search;
        let coarse_candidates = primary.coarse_centroid_candidates;
        let fine_candidates = primary.fine_centroid_candidates;
        let total_units = primary.total_units();

        primary_list_candidates.record(cost.primary_list_candidates);
        coarse_centroid_candidates.record(coarse_candidates);
        fine_centroid_candidates.record(fine_candidates);
        centroid_candidates.record(cost.centroid_candidates);
        centroid_early_discards.record(cost.centroid_early_discards);
        centroid_full_distance_candidates.record(cost.centroid_full_distance_candidates);
        centroid_vector_dimensions_evaluated.record(cost.centroid_vector_dimensions_evaluated);
        flat_candidates.record(cost.flat_candidates);
        flat_early_discards.record(cost.flat_early_discards);
        flat_full_distance_candidates.record(cost.flat_full_distance_candidates);
        flat_vector_dimensions_evaluated.record(cost.flat_vector_dimensions_evaluated);
        search_cost_units.record(total_units);

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
        hierarchy_build_elapsed_ms: references.hierarchy_build_elapsed_ms(),
        boundary_cases,
        boundary_case_percentage: percentage(boundary_cases, total),
        decision_mismatches,
        decision_mismatch_percentage: percentage(decision_mismatches, total),
        boundary_decision_mismatches,
        primary_only_decision_mismatches,
        avg_search_cost_units: search_cost_units.average(),
        max_search_cost_units: search_cost_units.max(),
    };
    let output = DiagnoseOutput {
        warnings: diagnose_warnings(&core),
        core,
        candidates: DiagnoseCandidates {
            avg_primary_list_candidates: primary_list_candidates.average(),
            max_primary_list_candidates: primary_list_candidates.max(),
            avg_flat_candidates: flat_candidates.average(),
            avg_flat_early_discards: flat_early_discards.average(),
            avg_flat_full_distance_candidates: flat_full_distance_candidates.average(),
            avg_flat_vector_dimensions_evaluated: flat_vector_dimensions_evaluated.average(),
        },
        centroids: DiagnoseCentroids {
            avg_coarse_centroid_candidates: coarse_centroid_candidates.average(),
            avg_fine_centroid_candidates: fine_centroid_candidates.average(),
            avg_centroid_candidates: centroid_candidates.average(),
            avg_centroid_early_discards: centroid_early_discards.average(),
            avg_centroid_full_distance_candidates: centroid_full_distance_candidates.average(),
            avg_centroid_vector_dimensions_evaluated: centroid_vector_dimensions_evaluated
                .average(),
        },
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
