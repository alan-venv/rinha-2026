use std::fs::File;
use std::time::Instant;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[path = "../consts.rs"]
mod consts;
#[path = "../dto.rs"]
mod dto;
#[path = "../memory.rs"]
mod memory;
#[path = "../service.rs"]
mod service;

use dto::ContentRequest;

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
    config: DiagnoseHierarchyConfig,
    hierarchy_build_elapsed_ms: u128,
    core: DiagnoseCore,
    candidates: DiagnoseCandidates,
    centroids: DiagnoseCentroids,
    avg_search_cost_units: usize,
    max_search_cost_units: usize,
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

#[derive(Serialize)]
struct DiagnoseHierarchyConfig {
    coarse_centroids: usize,
    coarse_probes: usize,
    coarse_iterations: usize,
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
    let mut primary_list_candidates = Vec::with_capacity(total);
    let mut coarse_centroid_candidates = Vec::with_capacity(total);
    let mut fine_centroid_candidates = Vec::with_capacity(total);
    let mut centroid_candidates = Vec::with_capacity(total);
    let mut centroid_early_discards = Vec::with_capacity(total);
    let mut centroid_full_distance_candidates = Vec::with_capacity(total);
    let mut centroid_vector_dimensions_evaluated = Vec::with_capacity(total);
    let mut flat_candidates = Vec::with_capacity(total);
    let mut flat_early_discards = Vec::with_capacity(total);
    let mut flat_full_distance_candidates = Vec::with_capacity(total);
    let mut flat_vector_dimensions_evaluated = Vec::with_capacity(total);
    let mut search_cost_units = Vec::with_capacity(total);

    for entry in data.entries {
        let vector = service::vectorization(entry.request);
        let cost = references.search_cost(&vector);
        primary_list_candidates.push(cost.search.primary_list_candidates);
        coarse_centroid_candidates.push(cost.coarse_centroid_candidates);
        fine_centroid_candidates.push(cost.fine_centroid_candidates);
        centroid_candidates.push(cost.search.centroid_candidates);
        centroid_early_discards.push(cost.search.centroid_early_discards);
        centroid_full_distance_candidates.push(cost.search.centroid_full_distance_candidates);
        centroid_vector_dimensions_evaluated.push(cost.search.centroid_vector_dimensions_evaluated);
        flat_candidates.push(cost.search.flat_candidates);
        flat_early_discards.push(cost.search.flat_early_discards);
        flat_full_distance_candidates.push(cost.search.flat_full_distance_candidates);
        flat_vector_dimensions_evaluated.push(cost.search.flat_vector_dimensions_evaluated);
        search_cost_units.push(cost.total_units());

        let details = service::fraud_score_details(&vector, &references);
        let score = details.score;
        let approved = score < 0.6;

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

    let core_elapsed_ms = started.elapsed().as_millis();
    let hierarchy_config = references.hierarchy_config();
    let output = DiagnoseOutput {
        config: DiagnoseHierarchyConfig {
            coarse_centroids: hierarchy_config.coarse_centroids,
            coarse_probes: hierarchy_config.coarse_probes,
            coarse_iterations: hierarchy_config.coarse_iterations,
        },
        hierarchy_build_elapsed_ms: references.hierarchy_build_elapsed_ms(),
        core: DiagnoseCore {
            entries: total,
            elapsed_ms: core_elapsed_ms,
            boundary_cases,
            boundary_case_percentage: percentage(boundary_cases, total),
            decision_mismatches,
            decision_mismatch_percentage: percentage(decision_mismatches, total),
            boundary_decision_mismatches,
            primary_only_decision_mismatches,
        },
        candidates: DiagnoseCandidates {
            avg_primary_list_candidates: average(&primary_list_candidates),
            max_primary_list_candidates: primary_list_candidates.iter().copied().max().unwrap_or(0),
            avg_flat_candidates: average(&flat_candidates),
            avg_flat_early_discards: average(&flat_early_discards),
            avg_flat_full_distance_candidates: average(&flat_full_distance_candidates),
            avg_flat_vector_dimensions_evaluated: average(&flat_vector_dimensions_evaluated),
        },
        centroids: DiagnoseCentroids {
            avg_coarse_centroid_candidates: average(&coarse_centroid_candidates),
            avg_fine_centroid_candidates: average(&fine_centroid_candidates),
            avg_centroid_candidates: average(&centroid_candidates),
            avg_centroid_early_discards: average(&centroid_early_discards),
            avg_centroid_full_distance_candidates: average(&centroid_full_distance_candidates),
            avg_centroid_vector_dimensions_evaluated: average(
                &centroid_vector_dimensions_evaluated,
            ),
        },
        avg_search_cost_units: average(&search_cost_units),
        max_search_cost_units: search_cost_units.iter().copied().max().unwrap_or(0),
    };

    println!("{}", serde_json::to_string_pretty(&output)?);
    print_warnings(&output);

    Ok(())
}

fn print_warnings(output: &DiagnoseOutput) {
    if output.core.elapsed_ms > MAX_ELAPSED_MS {
        println!(
            "warning: elapsed_ms={} > {}",
            output.core.elapsed_ms, MAX_ELAPSED_MS
        );
    }

    let boundary_case_percentage =
        percentage_value(output.core.boundary_cases, output.core.entries);
    if boundary_case_percentage > MAX_BOUNDARY_CASE_PERCENTAGE {
        println!(
            "warning: boundary_case_percentage={boundary_case_percentage:.2}% > {MAX_BOUNDARY_CASE_PERCENTAGE:.2}%"
        );
    }

    if output.core.primary_only_decision_mismatches > 0 {
        println!(
            "warning: primary_only_decision_mismatches={} > 0",
            output.core.primary_only_decision_mismatches
        );
    }
}

fn percentage(value: usize, total: usize) -> String {
    format!("{:.2}%", percentage_value(value, total))
}

fn average(values: &[usize]) -> usize {
    if values.is_empty() {
        return 0;
    }

    (values.iter().sum::<usize>() as f64 / values.len() as f64).round() as usize
}

fn percentage_value(value: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }

    ((value as f64 / total as f64) * 10_000.0).round() / 100.0
}
