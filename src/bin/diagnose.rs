use std::cell::Cell;
use std::fs::File;
use std::time::Instant;

use anyhow::Result;
use rinha::dto::ContentRequest;
use rinha::memory::{self, ReferenceSource};
use rinha::{encoding, service};
use serde::{Deserialize, Serialize};

const MAX_ELAPSED_MS: u128 = 6_000;
const MAX_BOUNDARY_CASE_PERCENTAGE: f64 = 10.0;
const TOP5_DISTRIBUTION_LEN: usize = 6;

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
    boundary_top_k: DiagnoseBoundaryTopK,
    boundary_fraud_progression: DiagnoseBoundaryFraudProgression,
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
struct DiagnoseBoundaryTopK {
    top_50: BoundaryTopKMetrics,
    top_100: BoundaryTopKMetrics,
    top_500: BoundaryTopKMetrics,
}

#[derive(Serialize)]
struct DiagnoseBoundaryFraudProgression {
    top5_fraud_counts: [usize; TOP5_DISTRIBUTION_LEN],
    top5_decision_mismatches: [usize; TOP5_DISTRIBUTION_LEN],
    top5_exactly_3: BoundaryGroupMetrics,
    top5_exactly_3_and_top100_exactly_60: BoundaryGroupMetrics,
    top5_exactly_3_and_top500_exactly_300: BoundaryGroupMetrics,
    top5_exactly_3_and_top100_exactly_60_and_top500_exactly_300: BoundaryGroupMetrics,
}

#[derive(Clone, Copy, Default, Serialize)]
struct BoundaryGroupMetrics {
    cases: usize,
    current_decision_mismatches: usize,
    expected_approved: usize,
    expected_rejected: usize,
    current_approved: usize,
    current_rejected: usize,
}

#[derive(Serialize)]
struct BoundaryTopKMetrics {
    evaluated: usize,
    decision_mismatches: usize,
    corrected_current_mismatches: usize,
    introduced_mismatches: usize,
    avg_fraud_count: usize,
    avg_fraud_percentage: String,
    current_match_cases: usize,
    current_match_avg_fraud_count: usize,
    current_match_avg_fraud_percentage: String,
    current_mismatch_cases: usize,
    current_mismatch_avg_fraud_count: usize,
    current_mismatch_avg_fraud_percentage: String,
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
    let mut boundary_top_50 = BoundaryTopKAccumulator::new(50);
    let mut boundary_top_100 = BoundaryTopKAccumulator::new(100);
    let mut boundary_top_500 = BoundaryTopKAccumulator::new(500);
    let mut boundary_progression = BoundaryFraudProgressionAccumulator::new();
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
            let current_mismatch = approved != entry.expected_approved;
            let top =
                top_primary_candidates::<500>(&references, &context, &vector, details.probes_used);

            boundary_top_50.record(
                &top.candidates[..top.len.min(50)],
                &references,
                entry.expected_approved,
                current_mismatch,
            );
            boundary_top_100.record(
                &top.candidates[..top.len.min(100)],
                &references,
                entry.expected_approved,
                current_mismatch,
            );
            boundary_top_500.record(
                &top.candidates[..top.len.min(500)],
                &references,
                entry.expected_approved,
                current_mismatch,
            );
            boundary_progression.record(
                &top.candidates[..top.len.min(500)],
                &references,
                entry.expected_approved,
                approved,
                current_mismatch,
            );
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
        boundary_top_k: DiagnoseBoundaryTopK {
            top_50: boundary_top_50.finish(),
            top_100: boundary_top_100.finish(),
            top_500: boundary_top_500.finish(),
        },
        boundary_fraud_progression: boundary_progression.finish(),
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

#[derive(Clone, Copy)]
struct TopCandidate {
    index: usize,
    distance: u64,
}

struct TopCandidates<const N: usize> {
    candidates: [TopCandidate; N],
    len: usize,
    farthest_slot: usize,
}

impl<const N: usize> TopCandidates<N> {
    fn new() -> Self {
        Self {
            candidates: [TopCandidate {
                index: 0,
                distance: u64::MAX,
            }; N],
            len: 0,
            farthest_slot: 0,
        }
    }

    fn current_worst_distance(&self) -> u64 {
        if self.len < N {
            u64::MAX
        } else {
            self.candidates[self.farthest_slot].distance
        }
    }

    fn add(&mut self, candidate: TopCandidate) {
        if self.len == N && candidate.distance >= self.current_worst_distance() {
            return;
        }

        if self.contains(candidate.index) {
            return;
        }

        if self.len < N {
            self.candidates[self.len] = candidate;
            self.len += 1;

            if candidate.distance > self.candidates[self.farthest_slot].distance {
                self.farthest_slot = self.len - 1;
            }

            return;
        }

        self.candidates[self.farthest_slot] = candidate;
        self.farthest_slot = self.find_farthest_slot();
    }

    fn sort_candidates(&mut self) {
        self.candidates[..self.len].sort_unstable_by_key(|candidate| candidate.distance);
        self.farthest_slot = self.find_farthest_slot();
    }

    fn contains(&self, index: usize) -> bool {
        self.candidates[..self.len]
            .iter()
            .any(|candidate| candidate.index == index)
    }

    fn find_farthest_slot(&self) -> usize {
        let mut farthest_slot = 0;

        for slot in 1..self.len {
            if self.candidates[slot].distance > self.candidates[farthest_slot].distance {
                farthest_slot = slot;
            }
        }

        farthest_slot
    }
}

fn top_primary_candidates<const N: usize>(
    references: &impl ReferenceSource,
    context: &memory::SearchContext,
    vector: &rinha::ReferenceVector,
    probe_count: usize,
) -> TopCandidates<N> {
    let mut top = TopCandidates::<N>::new();
    let current_worst_distance = Cell::new(u64::MAX);

    references.for_each_primary_candidate_batch(
        context,
        vector,
        0,
        probe_count,
        &mut || current_worst_distance.get(),
        &mut |index, distance| {
            top.add(TopCandidate { index, distance });
            current_worst_distance.set(top.current_worst_distance());
        },
    );
    top.sort_candidates();
    top
}

struct BoundaryTopKAccumulator {
    k: usize,
    evaluated: usize,
    decision_mismatches: usize,
    corrected_current_mismatches: usize,
    introduced_mismatches: usize,
    fraud_count_sum: usize,
    current_match_cases: usize,
    current_match_fraud_count_sum: usize,
    current_mismatch_cases: usize,
    current_mismatch_fraud_count_sum: usize,
}

impl BoundaryTopKAccumulator {
    fn new(k: usize) -> Self {
        Self {
            k,
            evaluated: 0,
            decision_mismatches: 0,
            corrected_current_mismatches: 0,
            introduced_mismatches: 0,
            fraud_count_sum: 0,
            current_match_cases: 0,
            current_match_fraud_count_sum: 0,
            current_mismatch_cases: 0,
            current_mismatch_fraud_count_sum: 0,
        }
    }

    fn record(
        &mut self,
        candidates: &[TopCandidate],
        references: &impl ReferenceSource,
        expected_approved: bool,
        current_mismatch: bool,
    ) {
        if candidates.is_empty() {
            return;
        }

        self.evaluated += 1;
        let fraud_count = candidates
            .iter()
            .filter(|candidate| references.is_fraud(candidate.index))
            .count();
        self.fraud_count_sum += fraud_count;

        if current_mismatch {
            self.current_mismatch_cases += 1;
            self.current_mismatch_fraud_count_sum += fraud_count;
        } else {
            self.current_match_cases += 1;
            self.current_match_fraud_count_sum += fraud_count;
        }

        let score = fraud_count as f32 / candidates.len() as f32;
        let approved = score < 0.6;
        let mismatch = approved != expected_approved;

        if mismatch {
            self.decision_mismatches += 1;
        }

        if current_mismatch && !mismatch {
            self.corrected_current_mismatches += 1;
        }

        if !current_mismatch && mismatch {
            self.introduced_mismatches += 1;
        }
    }

    fn finish(self) -> BoundaryTopKMetrics {
        BoundaryTopKMetrics {
            evaluated: self.evaluated,
            decision_mismatches: self.decision_mismatches,
            corrected_current_mismatches: self.corrected_current_mismatches,
            introduced_mismatches: self.introduced_mismatches,
            avg_fraud_count: if self.evaluated == 0 {
                0
            } else {
                self.fraud_count_sum / self.evaluated
            },
            avg_fraud_percentage: if self.evaluated == 0 {
                "0.00%".to_string()
            } else {
                percentage(self.fraud_count_sum, self.evaluated * self.k)
            },
            current_match_cases: self.current_match_cases,
            current_match_avg_fraud_count: average_count(
                self.current_match_fraud_count_sum,
                self.current_match_cases,
            ),
            current_match_avg_fraud_percentage: percentage_or_zero(
                self.current_match_fraud_count_sum,
                self.current_match_cases * self.k,
            ),
            current_mismatch_cases: self.current_mismatch_cases,
            current_mismatch_avg_fraud_count: average_count(
                self.current_mismatch_fraud_count_sum,
                self.current_mismatch_cases,
            ),
            current_mismatch_avg_fraud_percentage: percentage_or_zero(
                self.current_mismatch_fraud_count_sum,
                self.current_mismatch_cases * self.k,
            ),
        }
    }
}

fn average_count(sum: usize, count: usize) -> usize {
    if count == 0 { 0 } else { sum / count }
}

fn percentage_or_zero(count: usize, total: usize) -> String {
    if total == 0 {
        "0.00%".to_string()
    } else {
        percentage(count, total)
    }
}

struct BoundaryFraudProgressionAccumulator {
    top5_fraud_counts: [usize; TOP5_DISTRIBUTION_LEN],
    top5_decision_mismatches: [usize; TOP5_DISTRIBUTION_LEN],
    top5_exactly_3: BoundaryGroupAccumulator,
    top5_exactly_3_and_top100_exactly_60: BoundaryGroupAccumulator,
    top5_exactly_3_and_top500_exactly_300: BoundaryGroupAccumulator,
    top5_exactly_3_and_top100_exactly_60_and_top500_exactly_300: BoundaryGroupAccumulator,
}

impl BoundaryFraudProgressionAccumulator {
    fn new() -> Self {
        Self {
            top5_fraud_counts: [0; TOP5_DISTRIBUTION_LEN],
            top5_decision_mismatches: [0; TOP5_DISTRIBUTION_LEN],
            top5_exactly_3: BoundaryGroupAccumulator::default(),
            top5_exactly_3_and_top100_exactly_60: BoundaryGroupAccumulator::default(),
            top5_exactly_3_and_top500_exactly_300: BoundaryGroupAccumulator::default(),
            top5_exactly_3_and_top100_exactly_60_and_top500_exactly_300:
                BoundaryGroupAccumulator::default(),
        }
    }

    fn record(
        &mut self,
        candidates: &[TopCandidate],
        references: &impl ReferenceSource,
        expected_approved: bool,
        current_approved: bool,
        current_mismatch: bool,
    ) {
        let top5_fraud_count = fraud_count_in(candidates, 5, references);
        self.top5_fraud_counts[top5_fraud_count] += 1;
        if current_mismatch {
            self.top5_decision_mismatches[top5_fraud_count] += 1;
        }

        if top5_fraud_count != 3 {
            return;
        }

        let top100_fraud_count = fraud_count_in(candidates, 100, references);
        let top500_fraud_count = fraud_count_in(candidates, 500, references);
        let top100_exactly_60 = top100_fraud_count == 60;
        let top500_exactly_300 = top500_fraud_count == 300;

        self.top5_exactly_3
            .record(expected_approved, current_approved, current_mismatch);

        if top100_exactly_60 {
            self.top5_exactly_3_and_top100_exactly_60.record(
                expected_approved,
                current_approved,
                current_mismatch,
            );
        }

        if top500_exactly_300 {
            self.top5_exactly_3_and_top500_exactly_300.record(
                expected_approved,
                current_approved,
                current_mismatch,
            );
        }

        if top100_exactly_60 && top500_exactly_300 {
            self.top5_exactly_3_and_top100_exactly_60_and_top500_exactly_300
                .record(expected_approved, current_approved, current_mismatch);
        }
    }

    fn finish(self) -> DiagnoseBoundaryFraudProgression {
        DiagnoseBoundaryFraudProgression {
            top5_fraud_counts: self.top5_fraud_counts,
            top5_decision_mismatches: self.top5_decision_mismatches,
            top5_exactly_3: self.top5_exactly_3.finish(),
            top5_exactly_3_and_top100_exactly_60: self
                .top5_exactly_3_and_top100_exactly_60
                .finish(),
            top5_exactly_3_and_top500_exactly_300: self
                .top5_exactly_3_and_top500_exactly_300
                .finish(),
            top5_exactly_3_and_top100_exactly_60_and_top500_exactly_300: self
                .top5_exactly_3_and_top100_exactly_60_and_top500_exactly_300
                .finish(),
        }
    }
}

#[derive(Default)]
struct BoundaryGroupAccumulator {
    cases: usize,
    current_decision_mismatches: usize,
    expected_approved: usize,
    expected_rejected: usize,
    current_approved: usize,
    current_rejected: usize,
}

impl BoundaryGroupAccumulator {
    fn record(&mut self, expected_approved: bool, current_approved: bool, current_mismatch: bool) {
        self.cases += 1;

        if current_mismatch {
            self.current_decision_mismatches += 1;
        }

        if expected_approved {
            self.expected_approved += 1;
        } else {
            self.expected_rejected += 1;
        }

        if current_approved {
            self.current_approved += 1;
        } else {
            self.current_rejected += 1;
        }
    }

    fn finish(self) -> BoundaryGroupMetrics {
        BoundaryGroupMetrics {
            cases: self.cases,
            current_decision_mismatches: self.current_decision_mismatches,
            expected_approved: self.expected_approved,
            expected_rejected: self.expected_rejected,
            current_approved: self.current_approved,
            current_rejected: self.current_rejected,
        }
    }
}

fn fraud_count_in(
    candidates: &[TopCandidate],
    limit: usize,
    references: &impl ReferenceSource,
) -> usize {
    candidates
        .iter()
        .take(limit)
        .filter(|candidate| references.is_fraud(candidate.index))
        .count()
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
