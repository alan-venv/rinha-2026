use std::fs::File;
use std::time::Instant;

use anyhow::Result;
use memmap2::Mmap;
use rinha::{
    encoding,
    morton::MortonIndex,
    parser,
    service::{DecisionSource, Service},
};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

const DIMENSIONS: usize = 14;
const EXACT_K: usize = 5;
const KD_LEAF_SIZE: usize = 64;
const NO_CHILD: usize = usize::MAX;
const FLAT_CHECK_BOUNDARIES: usize = 32;

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
    #[serde(skip_serializing_if = "Option::is_none")]
    kd_tree: Option<Vec<KdTreeDiagnose>>,
}

#[derive(Serialize)]
struct KdTreeDiagnose {
    metric: &'static str,
    load_ms: u128,
    build_ms: u128,
    elapsed_ms: u128,
    nodes: usize,
    leaf_size: usize,
    boundary_cases: usize,
    boundary_mismatches: usize,
    comparisons_total: u64,
    comparisons_avg: u64,
    comparisons_p95: u32,
    comparisons_p99: u32,
    comparisons_max: u32,
    flat_checked: usize,
    flat_kd_mismatches: usize,
    flat_decision_mismatches: usize,
}

#[derive(Deserialize)]
struct ReferenceJson {
    vector: [f32; DIMENSIONS],
    label: JsonLabel,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum JsonLabel {
    Legit,
    Fraud,
}

struct ReferenceDataset {
    vectors: Vec<[i16; DIMENSIONS]>,
    labels: Vec<u8>,
}

struct Case {
    vector: [i16; DIMENSIONS],
    expected_fraud: bool,
    boundary: bool,
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
    let mut cases = Vec::with_capacity(total);

    for entry in data.entries {
        let request = parser::parse(entry.request.get().as_bytes())?;
        let vector = encoding::vectorization(&request);
        let decision = service.decide(&vector);
        let boundary = matches!(&decision.source, DecisionSource::Boundary);
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

        cases.push(Case {
            vector,
            expected_fraud: !entry.expected_approved,
            boundary,
        });
    }

    let kd_tree = if std::env::var("DIAGNOSE_KDTREE").as_deref() == Ok("1") {
        Some(diagnose_kd_tree(&cases, boundary_cases)?)
    } else {
        None
    };

    let diagnose = Diagnose {
        entries: total,
        elapsed_ms: started.elapsed().as_millis(),
        decision_mismatches,
        boundary_cases,
        morton_mismatches,
        kd_tree,
    };

    println!("{}", serde_json::to_string_pretty(&diagnose)?);

    Ok(())
}

impl ReferenceDataset {
    fn load() -> Result<Self> {
        let input = File::open("resources/references/references.json")?;
        let mmap = unsafe { Mmap::map(&input)? };
        let records: Vec<ReferenceJson> = serde_json::from_slice(&mmap)?;
        let mut vectors = Vec::with_capacity(records.len());
        let mut labels = Vec::with_capacity(records.len());

        for record in records {
            vectors.push(quantize_reference_vector(&record.vector));
            labels.push(match record.label {
                JsonLabel::Legit => 0,
                JsonLabel::Fraud => 1,
            });
        }

        Ok(Self { vectors, labels })
    }
}

struct KdTree {
    nodes: Vec<KdNode>,
    indices: Vec<u32>,
}

struct KdNode {
    min: [i16; DIMENSIONS],
    max: [i16; DIMENSIONS],
    left: usize,
    right: usize,
    start: usize,
    len: usize,
}

#[derive(Clone, Copy)]
struct Neighbor {
    distance: u64,
    index: u32,
    label: u8,
}

struct KdSearch {
    top: [Neighbor; EXACT_K],
    comparisons: u32,
}

impl KdTree {
    fn build(references: &ReferenceDataset) -> Self {
        let mut tree = Self {
            nodes: Vec::new(),
            indices: (0..references.vectors.len() as u32).collect(),
        };
        let len = tree.indices.len();
        if len > 0 {
            tree.build_node(references, 0, len);
        }
        tree
    }

    fn build_node(&mut self, references: &ReferenceDataset, start: usize, len: usize) -> usize {
        let (min, max) = bounds(references, &self.indices[start..start + len]);
        let node_index = self.nodes.len();
        self.nodes.push(KdNode {
            min,
            max,
            left: NO_CHILD,
            right: NO_CHILD,
            start,
            len,
        });

        if len <= KD_LEAF_SIZE {
            return node_index;
        }

        let split_dimension = widest_dimension(&min, &max);
        let middle = len / 2;
        self.indices[start..start + len].select_nth_unstable_by_key(middle, |index| {
            references.vectors[*index as usize][split_dimension]
        });

        let left = self.build_node(references, start, middle);
        let right = self.build_node(references, start + middle, len - middle);
        self.nodes[node_index].left = left;
        self.nodes[node_index].right = right;

        node_index
    }

    fn top5(
        &self,
        references: &ReferenceDataset,
        vector: &[i16; DIMENSIONS],
        metric: DistanceMetric,
    ) -> KdSearch {
        let mut search = KdSearch {
            top: [Neighbor {
                distance: u64::MAX,
                index: u32::MAX,
                label: 0,
            }; EXACT_K],
            comparisons: 0,
        };

        if self.nodes.is_empty() {
            return search;
        }

        let mut stack = vec![0_usize];
        while let Some(node_index) = stack.pop() {
            let node = &self.nodes[node_index];
            if lower_bound(vector, node, metric) > search.top[EXACT_K - 1].distance {
                continue;
            }

            if node.left == NO_CHILD {
                for &reference_index in &self.indices[node.start..node.start + node.len] {
                    let reference_index = reference_index as usize;
                    let distance =
                        distance_between(vector, &references.vectors[reference_index], metric);
                    search.comparisons += 1;
                    insert_neighbor(
                        &mut search.top,
                        Neighbor {
                            distance,
                            index: reference_index as u32,
                            label: references.labels[reference_index],
                        },
                    );
                }
                continue;
            }

            let left_bound = lower_bound(vector, &self.nodes[node.left], metric);
            let right_bound = lower_bound(vector, &self.nodes[node.right], metric);
            let worst = search.top[EXACT_K - 1].distance;

            if left_bound <= right_bound {
                if right_bound <= worst {
                    stack.push(node.right);
                }
                if left_bound <= worst {
                    stack.push(node.left);
                }
            } else {
                if left_bound <= worst {
                    stack.push(node.left);
                }
                if right_bound <= worst {
                    stack.push(node.right);
                }
            }
        }

        search
    }
}

#[derive(Clone, Copy)]
enum DistanceMetric {
    L1,
    L2,
}

impl DistanceMetric {
    fn name(self) -> &'static str {
        match self {
            Self::L1 => "l1",
            Self::L2 => "l2_squared",
        }
    }
}

fn diagnose_kd_tree(cases: &[Case], boundary_cases: usize) -> Result<Vec<KdTreeDiagnose>> {
    let load_started = Instant::now();
    let references = ReferenceDataset::load()?;
    let load_ms = load_started.elapsed().as_millis();

    let build_started = Instant::now();
    let tree = KdTree::build(&references);
    let build_ms = build_started.elapsed().as_millis();
    let mut outputs = Vec::new();

    for metric in [DistanceMetric::L1, DistanceMetric::L2] {
        outputs.push(diagnose_kd_metric(
            cases,
            boundary_cases,
            &references,
            &tree,
            metric,
            load_ms,
            build_ms,
        ));
    }

    Ok(outputs)
}

fn diagnose_kd_metric(
    cases: &[Case],
    boundary_cases: usize,
    references: &ReferenceDataset,
    tree: &KdTree,
    metric: DistanceMetric,
    load_ms: u128,
    build_ms: u128,
) -> KdTreeDiagnose {
    let started = Instant::now();
    let mut boundary_mismatches = 0;
    let mut comparisons = Vec::with_capacity(boundary_cases);
    let mut flat_checked = 0;
    let mut flat_kd_mismatches = 0;
    let mut flat_decision_mismatches = 0;

    for case in cases.iter().filter(|case| case.boundary) {
        let search = tree.top5(references, &case.vector, metric);
        let frauds = search
            .top
            .iter()
            .filter(|neighbor| neighbor.label == 1)
            .count();
        let predicted_fraud = frauds >= 3;
        boundary_mismatches += usize::from(predicted_fraud != case.expected_fraud);
        comparisons.push(search.comparisons);

        if flat_checked < FLAT_CHECK_BOUNDARIES {
            let flat = flat_top5(references, &case.vector, metric);
            flat_kd_mismatches += usize::from(!same_top(&search.top, &flat));

            let flat_frauds = flat.iter().filter(|neighbor| neighbor.label == 1).count();
            let flat_predicted_fraud = flat_frauds >= 3;
            flat_decision_mismatches += usize::from(flat_predicted_fraud != case.expected_fraud);
            flat_checked += 1;
        }
    }

    comparisons.sort_unstable();
    let comparisons_total = comparisons.iter().map(|value| *value as u64).sum::<u64>();
    let comparisons_avg = comparisons_total / boundary_cases.max(1) as u64;

    KdTreeDiagnose {
        metric: metric.name(),
        load_ms,
        build_ms,
        elapsed_ms: started.elapsed().as_millis(),
        nodes: tree.nodes.len(),
        leaf_size: KD_LEAF_SIZE,
        boundary_cases,
        boundary_mismatches,
        comparisons_total,
        comparisons_avg,
        comparisons_p95: percentile(&comparisons, 95),
        comparisons_p99: percentile(&comparisons, 99),
        comparisons_max: comparisons.last().copied().unwrap_or(0),
        flat_checked,
        flat_kd_mismatches,
        flat_decision_mismatches,
    }
}

fn flat_top5(
    references: &ReferenceDataset,
    vector: &[i16; DIMENSIONS],
    metric: DistanceMetric,
) -> [Neighbor; EXACT_K] {
    let mut top = [Neighbor {
        distance: u64::MAX,
        index: u32::MAX,
        label: 0,
    }; EXACT_K];

    for (index, reference) in references.vectors.iter().enumerate() {
        insert_neighbor(
            &mut top,
            Neighbor {
                distance: distance_between(vector, reference, metric),
                index: index as u32,
                label: references.labels[index],
            },
        );
    }

    top
}

fn same_top(left: &[Neighbor; EXACT_K], right: &[Neighbor; EXACT_K]) -> bool {
    left.iter()
        .zip(right)
        .all(|(left, right)| left.distance == right.distance && left.index == right.index)
}

fn bounds(
    references: &ReferenceDataset,
    indices: &[u32],
) -> ([i16; DIMENSIONS], [i16; DIMENSIONS]) {
    let mut min = [i16::MAX; DIMENSIONS];
    let mut max = [i16::MIN; DIMENSIONS];

    for &index in indices {
        let vector = &references.vectors[index as usize];
        for dimension in 0..DIMENSIONS {
            min[dimension] = min[dimension].min(vector[dimension]);
            max[dimension] = max[dimension].max(vector[dimension]);
        }
    }

    (min, max)
}

fn widest_dimension(min: &[i16; DIMENSIONS], max: &[i16; DIMENSIONS]) -> usize {
    let mut dimension = 0;
    let mut width = i32::MIN;

    for index in 0..DIMENSIONS {
        let candidate_width = max[index] as i32 - min[index] as i32;
        if candidate_width > width {
            width = candidate_width;
            dimension = index;
        }
    }

    dimension
}

fn lower_bound(vector: &[i16; DIMENSIONS], node: &KdNode, metric: DistanceMetric) -> u64 {
    let mut distance = 0_u64;

    for (dimension, value) in vector.iter().enumerate() {
        let difference = if *value < node.min[dimension] {
            node.min[dimension] as i32 - *value as i32
        } else if *value > node.max[dimension] {
            *value as i32 - node.max[dimension] as i32
        } else {
            0
        } as u64;

        distance += match metric {
            DistanceMetric::L1 => difference,
            DistanceMetric::L2 => difference * difference,
        };
    }

    distance
}

fn insert_neighbor(top: &mut [Neighbor; EXACT_K], neighbor: Neighbor) {
    if !neighbor_better(neighbor, top[EXACT_K - 1]) {
        return;
    }

    let mut output = EXACT_K - 1;
    while output > 0 && neighbor_better(neighbor, top[output - 1]) {
        top[output] = top[output - 1];
        output -= 1;
    }
    top[output] = neighbor;
}

fn neighbor_better(left: Neighbor, right: Neighbor) -> bool {
    left.distance < right.distance || (left.distance == right.distance && left.index < right.index)
}

fn quantize_reference_vector(vector: &[f32; DIMENSIONS]) -> [i16; DIMENSIONS] {
    let mut quantized = [0; DIMENSIONS];
    for (output, input) in quantized.iter_mut().zip(vector) {
        *output = (*input * 10_000.0).round() as i16;
    }
    quantized
}

fn distance_between(
    left: &[i16; DIMENSIONS],
    right: &[i16; DIMENSIONS],
    metric: DistanceMetric,
) -> u64 {
    let mut distance = 0_u64;

    for (left, right) in left.iter().zip(right) {
        let difference = (*left as i32 - *right as i32).unsigned_abs() as u64;
        distance += match metric {
            DistanceMetric::L1 => difference,
            DistanceMetric::L2 => difference * difference,
        };
    }

    distance
}

fn percentile(values: &[u32], percentile: usize) -> u32 {
    if values.is_empty() {
        return 0;
    }

    let index = ((values.len() - 1) * percentile) / 100;
    values[index]
}
