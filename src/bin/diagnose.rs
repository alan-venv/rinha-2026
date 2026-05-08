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
    config: DiagnoseHierarchyConfig,
    core: DiagnoseCore,
    candidates: DiagnoseCandidates,
    centroids: DiagnoseCentroids,
}

#[derive(Serialize)]
struct DiagnoseHierarchyConfig {
    coarse_centroids: usize,
    coarse_probes: usize,
    coarse_iterations: usize,
    boundary_coarse_group_probes: usize,
    boundary_coarse_group_fine_probes: usize,
}

#[derive(Serialize)]
struct DiagnoseCore {
    entries: usize,
    elapsed_ms: u128,
    hierarchy_build_elapsed_ms: u128,
    boundary_cases: usize,
    boundary_case_percentage: String,
    boundary_fallbacks: usize,
    boundary_rescue_fallbacks: usize,
    avg_probes_used: usize,
    max_probes_used: usize,
    decision_mismatches: usize,
    decision_mismatch_percentage: String,
    boundary_decision_mismatches: usize,
    primary_only_decision_mismatches: usize,
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
    let mut boundary_fallbacks = 0;
    let mut boundary_rescue_fallbacks = 0;
    let mut probes_used = Vec::with_capacity(total);
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
        let vector = encoding::vectorization(entry.request);
        let context = references.prepare_search_context(&vector);
        let details = service::fraud_score_details(&vector, &references);
        let primary = references.search_cost_for_probe_count_with_context(
            &context,
            &vector,
            rinha::IVF_INITIAL_PROBES,
        );
        let (cost, coarse_candidates, fine_candidates, total_units) = if details.fallback_used {
            let fallback = references.boundary_fallback_search_cost_with_context(&context, &vector);
            let primary_with_fallback = add_search_cost(primary.search, fallback.search);
            let primary_total_units = primary.total_units() + fallback.total_units();

            if details.rescue_used {
                let rescue = references.boundary_rescue_search_cost_with_context(&context, &vector);
                let cost = add_search_cost(primary_with_fallback, rescue.search);
                let total_units = primary_total_units + rescue.total_units();
                (
                    cost,
                    primary.coarse_centroid_candidates
                        + fallback.coarse_centroid_candidates
                        + rescue.coarse_centroid_candidates,
                    primary.fine_centroid_candidates
                        + fallback.fine_centroid_candidates
                        + rescue.fine_centroid_candidates,
                    total_units,
                )
            } else {
                (
                    primary_with_fallback,
                    primary.coarse_centroid_candidates + fallback.coarse_centroid_candidates,
                    primary.fine_centroid_candidates + fallback.fine_centroid_candidates,
                    primary_total_units,
                )
            }
        } else {
            (
                primary.search,
                primary.coarse_centroid_candidates,
                primary.fine_centroid_candidates,
                primary.total_units(),
            )
        };

        primary_list_candidates.push(cost.primary_list_candidates);
        coarse_centroid_candidates.push(coarse_candidates);
        fine_centroid_candidates.push(fine_candidates);
        centroid_candidates.push(cost.centroid_candidates);
        centroid_early_discards.push(cost.centroid_early_discards);
        centroid_full_distance_candidates.push(cost.centroid_full_distance_candidates);
        centroid_vector_dimensions_evaluated.push(cost.centroid_vector_dimensions_evaluated);
        flat_candidates.push(cost.flat_candidates);
        flat_early_discards.push(cost.flat_early_discards);
        flat_full_distance_candidates.push(cost.flat_full_distance_candidates);
        flat_vector_dimensions_evaluated.push(cost.flat_vector_dimensions_evaluated);
        search_cost_units.push(total_units);

        let approved = details.score < 0.6;
        probes_used.push(details.probes_used);

        if details.fallback_used {
            boundary_fallbacks += 1;
        }

        if details.rescue_used {
            boundary_rescue_fallbacks += 1;
        }

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

    let hierarchy_config = references.hierarchy_config();
    let output = DiagnoseOutput {
        config: DiagnoseHierarchyConfig {
            coarse_centroids: hierarchy_config.coarse_centroids,
            coarse_probes: hierarchy_config.coarse_probes,
            coarse_iterations: hierarchy_config.coarse_iterations,
            boundary_coarse_group_probes: hierarchy_config.boundary_coarse_group_probes,
            boundary_coarse_group_fine_probes: hierarchy_config.boundary_coarse_group_fine_probes,
        },
        core: DiagnoseCore {
            entries: total,
            elapsed_ms: started.elapsed().as_millis(),
            hierarchy_build_elapsed_ms: references.hierarchy_build_elapsed_ms(),
            boundary_cases,
            boundary_case_percentage: percentage(boundary_cases, total),
            boundary_fallbacks,
            boundary_rescue_fallbacks,
            avg_probes_used: average(&probes_used),
            max_probes_used: probes_used.iter().copied().max().unwrap_or(0),
            decision_mismatches,
            decision_mismatch_percentage: percentage(decision_mismatches, total),
            boundary_decision_mismatches,
            primary_only_decision_mismatches,
            avg_search_cost_units: average(&search_cost_units),
            max_search_cost_units: search_cost_units.iter().copied().max().unwrap_or(0),
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
    };

    println!("{}", serde_json::to_string_pretty(&output)?);
    print_warnings(&output);

    Ok(())
}

fn add_search_cost(left: memory::SearchCost, right: memory::SearchCost) -> memory::SearchCost {
    memory::SearchCost {
        primary_list_candidates: left.primary_list_candidates + right.primary_list_candidates,
        centroid_candidates: left.centroid_candidates + right.centroid_candidates,
        centroid_early_discards: left.centroid_early_discards + right.centroid_early_discards,
        centroid_full_distance_candidates: left.centroid_full_distance_candidates
            + right.centroid_full_distance_candidates,
        centroid_vector_dimensions_evaluated: left.centroid_vector_dimensions_evaluated
            + right.centroid_vector_dimensions_evaluated,
        flat_candidates: left.flat_candidates + right.flat_candidates,
        flat_early_discards: left.flat_early_discards + right.flat_early_discards,
        flat_full_distance_candidates: left.flat_full_distance_candidates
            + right.flat_full_distance_candidates,
        flat_vector_dimensions_evaluated: left.flat_vector_dimensions_evaluated
            + right.flat_vector_dimensions_evaluated,
    }
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
