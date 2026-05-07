use std::env;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use rinha::*;

const IVF_PATH: &str = "resources/ivf.bin";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceRecord {
    pub vector: ReferenceVector,
    pub is_fraud: bool,
}

pub trait ReferenceSource {
    fn is_fraud(&self, index: usize) -> bool;

    fn max_primary_probe_count(&self) -> usize {
        IVF_INITIAL_PROBES
    }

    fn for_each_primary_candidate_batch<C, V>(
        &self,
        vector: &ReferenceVector,
        start_probe: usize,
        end_probe: usize,
        current_worst_top_distance: &mut C,
        visit: &mut V,
    ) where
        C: FnMut() -> u64,
        V: FnMut(usize, u64);
}

pub struct IndexedReferences {
    ivfs: IvfIndexes,
    hierarchy: CentroidHierarchy,
    hierarchy_build_elapsed_ms: u128,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchCost {
    pub primary_list_candidates: usize,
    pub centroid_candidates: usize,
    pub centroid_early_discards: usize,
    pub centroid_full_distance_candidates: usize,
    pub centroid_vector_dimensions_evaluated: usize,
    pub flat_candidates: usize,
    pub flat_early_discards: usize,
    pub flat_full_distance_candidates: usize,
    pub flat_vector_dimensions_evaluated: usize,
}

impl SearchCost {
    #[allow(dead_code)]
    pub fn total_units(&self) -> usize {
        self.centroid_vector_dimensions_evaluated + self.flat_vector_dimensions_evaluated
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct HierarchyConfig {
    pub coarse_centroids: usize,
    pub coarse_probes: usize,
    pub coarse_iterations: usize,
    pub boundary_probe_batch: usize,
    pub boundary_max_probes: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HierarchySearchCost {
    pub search: SearchCost,
    pub coarse_centroid_candidates: usize,
    pub fine_centroid_candidates: usize,
}

impl HierarchySearchCost {
    #[allow(dead_code)]
    pub fn total_units(&self) -> usize {
        self.search.total_units()
    }
}

#[allow(dead_code)]
pub struct CentroidHierarchy {
    coarse_centroids: Vec<ReferenceVector>,
    offsets: Vec<u32>,
    fine_centroids: Vec<u32>,
}

pub fn load_references() -> Result<IndexedReferences> {
    let ivfs = IvfIndexes::load()?;
    let hierarchy_build_started = Instant::now();
    let hierarchy = ivfs.primary.build_centroid_hierarchy();
    let hierarchy_build_elapsed_ms = hierarchy_build_started.elapsed().as_millis();

    Ok(IndexedReferences {
        ivfs,
        hierarchy,
        hierarchy_build_elapsed_ms,
    })
}

fn read_u32_at(bytes: &[u8], offset: usize) -> u32 {
    let mut value = [0; size_of::<u32>()];
    value.copy_from_slice(&bytes[offset..offset + size_of::<u32>()]);
    u32::from_le_bytes(value)
}

fn read_u64_at(bytes: &[u8], offset: usize) -> u64 {
    let mut value = [0; size_of::<u64>()];
    value.copy_from_slice(&bytes[offset..offset + size_of::<u64>()]);
    u64::from_le_bytes(value)
}

impl ReferenceSource for IndexedReferences {
    fn is_fraud(&self, index: usize) -> bool {
        self.ivfs.is_fraud(index)
    }

    fn max_primary_probe_count(&self) -> usize {
        BOUNDARY_MAX_PROBES
    }

    fn for_each_primary_candidate_batch<C, V>(
        &self,
        vector: &ReferenceVector,
        start_probe: usize,
        end_probe: usize,
        current_worst_top_distance: &mut C,
        visit: &mut V,
    ) where
        C: FnMut() -> u64,
        V: FnMut(usize, u64),
    {
        self.ivfs.primary.for_each_hierarchy_candidate_batch(
            &self.hierarchy,
            vector,
            start_probe,
            end_probe,
            current_worst_top_distance,
            visit,
        );
    }
}

impl IndexedReferences {
    #[allow(dead_code)]
    pub fn search_cost(&self, vector: &ReferenceVector) -> HierarchySearchCost {
        self.search_cost_for_probe_count(vector, IVF_INITIAL_PROBES)
    }

    #[allow(dead_code)]
    pub fn search_cost_for_probe_count(
        &self,
        vector: &ReferenceVector,
        probe_count: usize,
    ) -> HierarchySearchCost {
        self.ivfs
            .primary
            .hierarchy_search_cost(&self.hierarchy, vector, probe_count)
    }

    #[allow(dead_code)]
    pub fn hierarchy_config(&self) -> HierarchyConfig {
        HierarchyConfig {
            coarse_centroids: HIERARCHY_COARSE_CENTROIDS,
            coarse_probes: HIERARCHY_COARSE_PROBES,
            coarse_iterations: HIERARCHY_COARSE_ITERATIONS,
            boundary_probe_batch: BOUNDARY_PROBE_BATCH,
            boundary_max_probes: BOUNDARY_MAX_PROBES,
        }
    }

    #[allow(dead_code)]
    pub fn hierarchy_build_elapsed_ms(&self) -> u128 {
        self.hierarchy_build_elapsed_ms
    }
}

impl ReferenceSource for [ReferenceRecord] {
    fn is_fraud(&self, index: usize) -> bool {
        self[index].is_fraud
    }

    fn for_each_primary_candidate_batch<C, V>(
        &self,
        vector: &ReferenceVector,
        start_probe: usize,
        _end_probe: usize,
        current_worst_top_distance: &mut C,
        visit: &mut V,
    ) where
        C: FnMut() -> u64,
        V: FnMut(usize, u64),
    {
        if start_probe > 0 {
            return;
        }

        for index in 0..self.len() {
            let max_useful_distance = current_worst_top_distance();
            let distance = distance2_limited(vector, &self[index].vector, max_useful_distance);

            if is_candidate_distance_useful(distance, max_useful_distance) {
                visit(index, distance);
            }
        }
    }
}

impl<const N: usize> ReferenceSource for [ReferenceRecord; N] {
    fn is_fraud(&self, index: usize) -> bool {
        self[index].is_fraud
    }

    fn for_each_primary_candidate_batch<C, V>(
        &self,
        vector: &ReferenceVector,
        start_probe: usize,
        end_probe: usize,
        current_worst_top_distance: &mut C,
        visit: &mut V,
    ) where
        C: FnMut() -> u64,
        V: FnMut(usize, u64),
    {
        self.as_slice().for_each_primary_candidate_batch(
            vector,
            start_probe,
            end_probe,
            current_worst_top_distance,
            visit,
        );
    }
}

fn distance2_limited(a: &ReferenceVector, b: &ReferenceVector, limit: u64) -> u64 {
    let mut distance = 0;

    for (left, right) in a.iter().zip(b) {
        let delta = i64::from(*left) - i64::from(*right);
        distance += (delta * delta) as u64;

        if distance >= limit {
            return distance;
        }
    }

    distance
}

fn distance2_vector_limited(a: &ReferenceVector, b: &ReferenceVector, limit: u64) -> DistanceEval {
    let mut distance = distance2_first8_vector(a, b);

    if distance >= limit {
        return DistanceEval {
            distance,
            dimensions: SIMD_EARLY_DIMENSIONS,
            early_discard: true,
        };
    }

    let mut dimensions = SIMD_EARLY_DIMENSIONS;
    for (left, right) in a.iter().zip(b).skip(SIMD_EARLY_DIMENSIONS) {
        distance += distance2_scalar_delta(*left, *right);
        dimensions += 1;

        if distance >= limit {
            return DistanceEval {
                distance,
                dimensions,
                early_discard: false,
            };
        }
    }

    DistanceEval {
        distance,
        dimensions,
        early_discard: false,
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn distance2_first8_vector(a: &ReferenceVector, b: &ReferenceVector) -> u64 {
    use std::arch::x86_64::*;

    unsafe {
        let left = _mm_loadu_si128(a.as_ptr().cast::<__m128i>());
        let right = _mm_loadu_si128(b.as_ptr().cast::<__m128i>());
        distance2_first8_sse2(left, right)
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn distance2_first8_vector(a: &ReferenceVector, b: &ReferenceVector) -> u64 {
    a.iter()
        .zip(b)
        .take(SIMD_EARLY_DIMENSIONS)
        .map(|(left, right)| distance2_scalar_delta(*left, *right))
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn distance2_first8_mmap(reference: *const i16, vector: &ReferenceVector) -> u64 {
    use std::arch::x86_64::*;

    unsafe {
        let left = _mm_loadu_si128(vector.as_ptr().cast::<__m128i>());
        let right = _mm_loadu_si128(reference.cast::<__m128i>());
        distance2_first8_sse2(left, right)
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn distance2_first8_mmap(reference: *const i16, vector: &ReferenceVector) -> u64 {
    (0..SIMD_EARLY_DIMENSIONS)
        .map(|dimension| {
            let reference_value = i16::from_le(unsafe { *reference.add(dimension) });
            distance2_scalar_delta(vector[dimension], reference_value)
        })
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn distance2_first8_sse2(
    left: std::arch::x86_64::__m128i,
    right: std::arch::x86_64::__m128i,
) -> u64 {
    use std::arch::x86_64::*;

    let delta = unsafe { _mm_sub_epi16(left, right) };
    let products = unsafe { _mm_madd_epi16(delta, delta) };
    let mut lanes = [0_i32; 4];
    unsafe { _mm_storeu_si128(lanes.as_mut_ptr().cast::<__m128i>(), products) };

    lanes.iter().map(|value| *value as u64).sum()
}

#[inline(always)]
fn distance2_scalar_delta(left: i16, right: i16) -> u64 {
    let delta = i64::from(left) - i64::from(right);
    (delta * delta) as u64
}

fn initial_sampled_centroids(
    vectors: &[ReferenceVector],
    centroid_count: usize,
) -> Vec<ReferenceVector> {
    let mut centroids = Vec::with_capacity(centroid_count);

    for centroid in 0..centroid_count {
        let index = centroid * vectors.len() / centroid_count;
        centroids.push(vectors[index]);
    }

    centroids
}

fn nearest_vector_index(centroids: &[ReferenceVector], vector: &ReferenceVector) -> usize {
    let mut nearest = 0;
    let mut nearest_distance = u64::MAX;

    for (index, centroid) in centroids.iter().enumerate() {
        let distance = distance2_limited(vector, centroid, nearest_distance);

        if distance < nearest_distance {
            nearest = index;
            nearest_distance = distance;
        }
    }

    nearest
}

fn is_candidate_distance_useful(distance: u64, limit: u64) -> bool {
    limit == u64::MAX || distance < limit
}

#[derive(Clone, Copy)]
struct DistanceEval {
    distance: u64,
    dimensions: usize,
    early_discard: bool,
}

#[derive(Default)]
struct CentroidSearchCost {
    candidates: usize,
    early_discards: usize,
    full_distance_candidates: usize,
    vector_dimensions_evaluated: usize,
}

struct SelectedCentroids<const N: usize> {
    centroids: [usize; N],
    len: usize,
    cost: CentroidSearchCost,
    coarse_centroid_candidates: usize,
    fine_centroid_candidates: usize,
}

struct TopCentroids<const N: usize> {
    indexes: [usize; N],
    distances: [u64; N],
    len: usize,
    farthest_slot: usize,
}

impl<const N: usize> TopCentroids<N> {
    fn new() -> Self {
        Self {
            indexes: [0; N],
            distances: [u64::MAX; N],
            len: 0,
            farthest_slot: 0,
        }
    }

    fn current_worst_distance(&self) -> u64 {
        if self.len < N {
            u64::MAX
        } else {
            self.distances[self.farthest_slot]
        }
    }

    fn add(&mut self, index: usize, distance: u64) {
        if self.len == N && distance >= self.current_worst_distance() {
            return;
        }

        if self.len < N {
            self.indexes[self.len] = index;
            self.distances[self.len] = distance;

            if distance > self.distances[self.farthest_slot] {
                self.farthest_slot = self.len;
            }

            self.len += 1;
            return;
        }

        self.indexes[self.farthest_slot] = index;
        self.distances[self.farthest_slot] = distance;
        self.farthest_slot = farthest_slot_in(&self.distances);
    }

    fn indexes(&self) -> &[usize] {
        &self.indexes[..self.len]
    }

    fn sort_indexes(&mut self) {
        let indexes = &mut self.indexes;
        let distances = &mut self.distances;

        for index in 1..self.len {
            let mut current = index;

            while current > 0 && distances[current] < distances[current - 1] {
                distances.swap(current, current - 1);
                indexes.swap(current, current - 1);
                current -= 1;
            }
        }

        self.farthest_slot = self.len.saturating_sub(1);
    }
}

struct IvfIndex {
    mmap: Mmap,
    reference_count: usize,
    centroid_count: usize,
    centroids_offset: usize,
    offsets_offset: usize,
    indices_offset: usize,
    vectors_offset: usize,
    fraud_offset: usize,
}

struct IvfIndexes {
    primary: IvfIndex,
}

impl CentroidHierarchy {
    fn select_centroids<const N: usize>(
        &self,
        index: &IvfIndex,
        vector: &ReferenceVector,
    ) -> SelectedCentroids<N> {
        let mut coarse_nearest = TopCentroids::<HIERARCHY_COARSE_PROBES>::new();
        let mut fine_nearest = TopCentroids::<N>::new();
        let mut cost = CentroidSearchCost::default();
        let mut coarse_centroid_candidates = 0;
        let mut fine_centroid_candidates = 0;

        for (coarse, centroid) in self.coarse_centroids.iter().enumerate() {
            let limit = coarse_nearest.current_worst_distance();
            let evaluation = distance2_vector_limited(vector, centroid, limit);

            coarse_centroid_candidates += 1;
            cost.candidates += 1;
            cost.vector_dimensions_evaluated += evaluation.dimensions;

            if evaluation.early_discard {
                cost.early_discards += 1;
            } else {
                cost.full_distance_candidates += 1;
            }

            if is_candidate_distance_useful(evaluation.distance, limit) {
                coarse_nearest.add(coarse, evaluation.distance);
            }
        }

        for coarse in coarse_nearest.indexes().iter().copied() {
            let start = self.offsets[coarse] as usize;
            let end = self.offsets[coarse + 1] as usize;

            for position in start..end {
                let fine = self.fine_centroids[position] as usize;
                let limit = fine_nearest.current_worst_distance();
                let evaluation = index.centroid_distance2_at_limited(fine, vector, limit);

                fine_centroid_candidates += 1;
                cost.candidates += 1;
                cost.vector_dimensions_evaluated += evaluation.dimensions;

                if evaluation.early_discard {
                    cost.early_discards += 1;
                } else {
                    cost.full_distance_candidates += 1;
                }

                if is_candidate_distance_useful(evaluation.distance, limit) {
                    fine_nearest.add(fine, evaluation.distance);
                }
            }
        }

        fine_nearest.sort_indexes();

        SelectedCentroids {
            centroids: fine_nearest.indexes,
            len: fine_nearest.len,
            cost,
            coarse_centroid_candidates,
            fine_centroid_candidates,
        }
    }
}

impl IvfIndexes {
    fn load() -> Result<Self> {
        Ok(Self {
            primary: IvfIndex::load("IVF_PATH", IVF_PATH, None)?,
        })
    }

    fn is_fraud(&self, index: usize) -> bool {
        self.primary.is_fraud(index)
    }

    #[allow(dead_code)]
    fn for_each_primary_candidates<C, V>(
        &self,
        vector: &ReferenceVector,
        current_worst_top_distance: &mut C,
        visit: &mut V,
    ) where
        C: FnMut() -> u64,
        V: FnMut(usize, u64),
    {
        self.primary
            .for_each_candidates(vector, current_worst_top_distance, visit);
    }

    #[allow(dead_code)]
    fn search_cost(&self, vector: &ReferenceVector) -> SearchCost {
        self.primary.search_cost(vector)
    }
}

impl IvfIndex {
    fn load(env_key: &str, default_path: &str, reference_count: Option<usize>) -> Result<Self> {
        let path = env::var(env_key).unwrap_or_else(|_| default_path.to_string());
        let input_file = File::open(Path::new(&path))
            .with_context(|| format!("failed to open required IVF index at {path}"))?;
        let mmap = unsafe { Mmap::map(&input_file) }?;
        let layout = IvfLayout::read(&mmap, reference_count)?;

        Ok(Self {
            mmap,
            reference_count: layout.reference_count,
            centroid_count: layout.centroid_count,
            centroids_offset: layout.centroids_offset,
            offsets_offset: layout.offsets_offset,
            indices_offset: layout.indices_offset,
            vectors_offset: layout.vectors_offset,
            fraud_offset: layout.fraud_offset,
        })
    }

    fn is_fraud(&self, index: usize) -> bool {
        debug_assert!(index < self.reference_count);
        let byte = self.mmap[self.fraud_offset + index / 8];
        byte & (1 << (index % 8)) != 0
    }

    #[allow(dead_code)]
    fn for_each_candidates<C, V>(
        &self,
        vector: &ReferenceVector,
        current_worst_top_distance: &mut C,
        visit: &mut V,
    ) where
        C: FnMut() -> u64,
        V: FnMut(usize, u64),
    {
        let (centroids, centroid_count) = self.nearest_centroid_indexes(vector);

        self.for_each_candidates_from_centroids(
            &centroids[..centroid_count],
            vector,
            current_worst_top_distance,
            visit,
        );
    }

    fn for_each_hierarchy_candidate_batch<C, V>(
        &self,
        hierarchy: &CentroidHierarchy,
        vector: &ReferenceVector,
        start_probe: usize,
        end_probe: usize,
        current_worst_top_distance: &mut C,
        visit: &mut V,
    ) where
        C: FnMut() -> u64,
        V: FnMut(usize, u64),
    {
        let end_probe = end_probe.min(BOUNDARY_MAX_PROBES);

        if end_probe <= IVF_INITIAL_PROBES {
            self.for_each_selected_hierarchy_candidate_batch::<IVF_INITIAL_PROBES, _, _>(
                hierarchy,
                vector,
                start_probe,
                end_probe,
                current_worst_top_distance,
                visit,
            );
        } else if end_probe <= { IVF_INITIAL_PROBES + BOUNDARY_PROBE_BATCH } {
            self.for_each_selected_hierarchy_candidate_batch::<
                { IVF_INITIAL_PROBES + BOUNDARY_PROBE_BATCH },
                _,
                _,
            >(
                hierarchy,
                vector,
                start_probe,
                end_probe,
                current_worst_top_distance,
                visit,
            );
        } else if end_probe <= { IVF_INITIAL_PROBES + BOUNDARY_PROBE_BATCH * 2 } {
            self.for_each_selected_hierarchy_candidate_batch::<
                { IVF_INITIAL_PROBES + BOUNDARY_PROBE_BATCH * 2 },
                _,
                _,
            >(
                hierarchy,
                vector,
                start_probe,
                end_probe,
                current_worst_top_distance,
                visit,
            );
        } else {
            self.for_each_selected_hierarchy_candidate_batch::<BOUNDARY_MAX_PROBES, _, _>(
                hierarchy,
                vector,
                start_probe,
                end_probe,
                current_worst_top_distance,
                visit,
            );
        }
    }

    fn for_each_selected_hierarchy_candidate_batch<const N: usize, C, V>(
        &self,
        hierarchy: &CentroidHierarchy,
        vector: &ReferenceVector,
        start_probe: usize,
        end_probe: usize,
        current_worst_top_distance: &mut C,
        visit: &mut V,
    ) where
        C: FnMut() -> u64,
        V: FnMut(usize, u64),
    {
        let selection = hierarchy.select_centroids::<N>(self, vector);
        let start = start_probe.min(selection.len);
        let end = end_probe.min(selection.len);

        if start >= end {
            return;
        }

        self.for_each_candidates_from_centroids(
            &selection.centroids[start..end],
            vector,
            current_worst_top_distance,
            visit,
        );
    }

    fn for_each_candidates_from_centroids<C, V>(
        &self,
        centroids: &[usize],
        vector: &ReferenceVector,
        current_worst_top_distance: &mut C,
        visit: &mut V,
    ) where
        C: FnMut() -> u64,
        V: FnMut(usize, u64),
    {
        for centroid in centroids.iter().copied() {
            let start = self.candidate_list_boundary_at(centroid) as usize;
            let end = self.candidate_list_boundary_at(centroid + 1) as usize;

            for position in start..end {
                let max_useful_distance = current_worst_top_distance();
                let index = self.candidate_index_at(position) as usize;
                let distance = self
                    .reference_distance2_at_limited(index, vector, max_useful_distance)
                    .distance;

                if is_candidate_distance_useful(distance, max_useful_distance) {
                    visit(index, distance);
                }
            }
        }
    }

    #[allow(dead_code)]
    fn search_cost(&self, vector: &ReferenceVector) -> SearchCost {
        let (centroids, centroid_count, centroid_cost) =
            self.nearest_centroid_indexes_with_cost(vector);
        let mut cost = self.search_cost_from_centroids(&centroids[..centroid_count], vector);

        cost.centroid_candidates = centroid_cost.candidates;
        cost.centroid_early_discards = centroid_cost.early_discards;
        cost.centroid_full_distance_candidates = centroid_cost.full_distance_candidates;
        cost.centroid_vector_dimensions_evaluated = centroid_cost.vector_dimensions_evaluated;

        cost
    }

    #[allow(dead_code)]
    fn hierarchy_search_cost(
        &self,
        hierarchy: &CentroidHierarchy,
        vector: &ReferenceVector,
        probe_count: usize,
    ) -> HierarchySearchCost {
        let probe_count = probe_count.min(BOUNDARY_MAX_PROBES);

        if probe_count <= IVF_INITIAL_PROBES {
            return self.hierarchy_search_cost_for::<IVF_INITIAL_PROBES>(
                hierarchy,
                vector,
                probe_count,
            );
        }

        if probe_count <= { IVF_INITIAL_PROBES + BOUNDARY_PROBE_BATCH } {
            return self
                .hierarchy_search_cost_for::<{ IVF_INITIAL_PROBES + BOUNDARY_PROBE_BATCH }>(
                    hierarchy,
                    vector,
                    probe_count,
                );
        }

        if probe_count <= { IVF_INITIAL_PROBES + BOUNDARY_PROBE_BATCH * 2 } {
            return self
                .hierarchy_search_cost_for::<{ IVF_INITIAL_PROBES + BOUNDARY_PROBE_BATCH * 2 }>(
                    hierarchy,
                    vector,
                    probe_count,
                );
        }

        self.hierarchy_search_cost_for::<BOUNDARY_MAX_PROBES>(hierarchy, vector, probe_count)
    }

    #[allow(dead_code)]
    fn hierarchy_search_cost_for<const N: usize>(
        &self,
        hierarchy: &CentroidHierarchy,
        vector: &ReferenceVector,
        probe_count: usize,
    ) -> HierarchySearchCost {
        let selection = hierarchy.select_centroids::<N>(self, vector);
        let selected_len = probe_count.min(selection.len);
        let mut search =
            self.search_cost_from_centroids(&selection.centroids[..selected_len], vector);

        search.centroid_candidates = selection.cost.candidates;
        search.centroid_early_discards = selection.cost.early_discards;
        search.centroid_full_distance_candidates = selection.cost.full_distance_candidates;
        search.centroid_vector_dimensions_evaluated = selection.cost.vector_dimensions_evaluated;

        HierarchySearchCost {
            search,
            coarse_centroid_candidates: selection.coarse_centroid_candidates,
            fine_centroid_candidates: selection.fine_centroid_candidates,
        }
    }

    #[allow(dead_code)]
    fn search_cost_from_centroids(
        &self,
        centroids: &[usize],
        vector: &ReferenceVector,
    ) -> SearchCost {
        let primary_list_candidates = centroids
            .iter()
            .map(|centroid| {
                let start = self.candidate_list_boundary_at(*centroid) as usize;
                let end = self.candidate_list_boundary_at(*centroid + 1) as usize;
                end - start
            })
            .sum();
        let mut nearest = CostTop::new();
        let mut cost = SearchCost {
            primary_list_candidates,
            ..SearchCost::default()
        };

        for centroid in centroids.iter().copied() {
            let start = self.candidate_list_boundary_at(centroid) as usize;
            let end = self.candidate_list_boundary_at(centroid + 1) as usize;

            for position in start..end {
                let index = self.candidate_index_at(position) as usize;
                let limit = nearest.current_worst_distance();
                let evaluation = self.reference_distance2_at_limited(index, vector, limit);

                cost.flat_candidates += 1;
                cost.flat_vector_dimensions_evaluated += evaluation.dimensions;

                if evaluation.early_discard {
                    cost.flat_early_discards += 1;
                } else {
                    cost.flat_full_distance_candidates += 1;
                }

                if is_candidate_distance_useful(evaluation.distance, limit) {
                    nearest.add(index, evaluation.distance);
                }
            }
        }

        cost
    }

    #[allow(dead_code)]
    fn build_centroid_hierarchy(&self) -> CentroidHierarchy {
        let fine_centroid_vectors = (0..self.centroid_count)
            .map(|centroid| self.centroid_vector_at(centroid))
            .collect::<Vec<_>>();
        let coarse_count = HIERARCHY_COARSE_CENTROIDS
            .min(fine_centroid_vectors.len())
            .max(1);
        let mut coarse_centroids = initial_sampled_centroids(&fine_centroid_vectors, coarse_count);

        for _ in 0..HIERARCHY_COARSE_ITERATIONS {
            let mut sums = vec![[0_i64; VECTOR_DIMENSIONS]; coarse_count];
            let mut counts = vec![0_u32; coarse_count];

            for vector in &fine_centroid_vectors {
                let coarse = nearest_vector_index(&coarse_centroids, vector);
                counts[coarse] += 1;

                for (sum, value) in sums[coarse].iter_mut().zip(vector) {
                    *sum += i64::from(*value);
                }
            }

            for coarse in 0..coarse_count {
                let count = counts[coarse];

                if count == 0 {
                    continue;
                }

                for dimension in 0..VECTOR_DIMENSIONS {
                    coarse_centroids[coarse][dimension] =
                        (sums[coarse][dimension] / i64::from(count)) as i16;
                }
            }
        }

        let mut assignments = Vec::with_capacity(fine_centroid_vectors.len());
        let mut counts = vec![0_u32; coarse_count];

        for vector in &fine_centroid_vectors {
            let coarse = nearest_vector_index(&coarse_centroids, vector);
            assignments.push(coarse);
            counts[coarse] += 1;
        }

        let mut offsets = vec![0_u32; coarse_count + 1];
        for coarse in 0..coarse_count {
            offsets[coarse + 1] = offsets[coarse] + counts[coarse];
        }

        let mut cursors = offsets[..coarse_count].to_vec();
        let mut fine_centroids = vec![0_u32; fine_centroid_vectors.len()];

        for (fine, coarse) in assignments.into_iter().enumerate() {
            let cursor = &mut cursors[coarse];
            fine_centroids[*cursor as usize] = fine as u32;
            *cursor += 1;
        }

        CentroidHierarchy {
            coarse_centroids,
            offsets,
            fine_centroids,
        }
    }

    #[allow(dead_code)]
    fn nearest_centroid_indexes(
        &self,
        vector: &ReferenceVector,
    ) -> ([usize; IVF_INITIAL_PROBES], usize) {
        let mut nearest = [0; IVF_INITIAL_PROBES];
        let mut distances = [u64::MAX; IVF_INITIAL_PROBES];
        let mut len = 0;
        let mut farthest_slot = 0;

        for centroid in 0..self.centroid_count {
            let limit = if len < IVF_INITIAL_PROBES {
                u64::MAX
            } else {
                distances[farthest_slot]
            };
            let distance = self
                .centroid_distance2_at_limited(centroid, vector, limit)
                .distance;

            if len == IVF_INITIAL_PROBES && distance >= distances[farthest_slot] {
                continue;
            }

            if len < IVF_INITIAL_PROBES {
                nearest[len] = centroid;
                distances[len] = distance;

                if distance > distances[farthest_slot] {
                    farthest_slot = len;
                }

                len += 1;
                continue;
            }

            nearest[farthest_slot] = centroid;
            distances[farthest_slot] = distance;
            farthest_slot = farthest_slot_in(&distances);
        }

        (nearest, len)
    }

    #[allow(dead_code)]
    fn nearest_centroid_indexes_with_cost(
        &self,
        vector: &ReferenceVector,
    ) -> ([usize; IVF_INITIAL_PROBES], usize, CentroidSearchCost) {
        let mut nearest = [0; IVF_INITIAL_PROBES];
        let mut distances = [u64::MAX; IVF_INITIAL_PROBES];
        let mut len = 0;
        let mut farthest_slot = 0;
        let mut cost = CentroidSearchCost::default();

        for centroid in 0..self.centroid_count {
            let limit = if len < IVF_INITIAL_PROBES {
                u64::MAX
            } else {
                distances[farthest_slot]
            };
            let evaluation = self.centroid_distance2_at_limited(centroid, vector, limit);
            let distance = evaluation.distance;

            cost.candidates += 1;
            cost.vector_dimensions_evaluated += evaluation.dimensions;

            if evaluation.early_discard {
                cost.early_discards += 1;
            } else {
                cost.full_distance_candidates += 1;
            }

            if len == IVF_INITIAL_PROBES && distance >= distances[farthest_slot] {
                continue;
            }

            if len < IVF_INITIAL_PROBES {
                nearest[len] = centroid;
                distances[len] = distance;

                if distance > distances[farthest_slot] {
                    farthest_slot = len;
                }

                len += 1;
                continue;
            }

            nearest[farthest_slot] = centroid;
            distances[farthest_slot] = distance;
            farthest_slot = farthest_slot_in(&distances);
        }

        (nearest, len, cost)
    }

    fn centroid_distance2_at_limited(
        &self,
        centroid: usize,
        vector: &ReferenceVector,
        limit: u64,
    ) -> DistanceEval {
        distance2_mmap_at_limited(
            &self.mmap,
            self.centroid_vector_offset(centroid),
            vector,
            limit,
        )
    }

    fn reference_distance2_at_limited(
        &self,
        index: usize,
        vector: &ReferenceVector,
        limit: u64,
    ) -> DistanceEval {
        distance2_mmap_at_limited(
            &self.mmap,
            self.reference_vector_offset(index),
            vector,
            limit,
        )
    }

    fn candidate_list_boundary_at(&self, centroid: usize) -> u32 {
        read_u32_at(&self.mmap, self.candidate_list_boundary_offset(centroid))
    }

    fn candidate_index_at(&self, position: usize) -> u32 {
        read_u32_at(&self.mmap, self.candidate_index_offset(position))
    }

    fn centroid_vector_at(&self, centroid: usize) -> ReferenceVector {
        vector_mmap_at(&self.mmap, self.centroid_vector_offset(centroid))
    }

    fn centroid_vector_offset(&self, centroid: usize) -> usize {
        self.centroids_offset + centroid * VECTOR_LEN
    }

    fn candidate_list_boundary_offset(&self, centroid: usize) -> usize {
        self.offsets_offset + centroid * size_of::<u32>()
    }

    fn candidate_index_offset(&self, position: usize) -> usize {
        self.indices_offset + position * size_of::<u32>()
    }

    fn reference_vector_offset(&self, index: usize) -> usize {
        self.vectors_offset + index * VECTOR_LEN
    }
}

fn distance2_mmap_at_limited(
    mmap: &Mmap,
    offset: usize,
    vector: &ReferenceVector,
    limit: u64,
) -> DistanceEval {
    let reference = unsafe { mmap.as_ptr().add(offset).cast::<i16>() };
    let mut distance = distance2_first8_mmap(reference, vector);

    if distance >= limit {
        return DistanceEval {
            distance,
            dimensions: SIMD_EARLY_DIMENSIONS,
            early_discard: true,
        };
    }

    let mut dimensions = SIMD_EARLY_DIMENSIONS;
    for (dimension, query_value) in vector.iter().enumerate().skip(SIMD_EARLY_DIMENSIONS) {
        let reference_value = i16::from_le(unsafe { *reference.add(dimension) });
        distance += distance2_scalar_delta(*query_value, reference_value);
        dimensions += 1;

        if distance >= limit {
            return DistanceEval {
                distance,
                dimensions,
                early_discard: false,
            };
        }
    }

    DistanceEval {
        distance,
        dimensions,
        early_discard: false,
    }
}

fn vector_mmap_at(mmap: &Mmap, offset: usize) -> ReferenceVector {
    let reference = unsafe { mmap.as_ptr().add(offset).cast::<i16>() };
    let mut vector = [0; VECTOR_DIMENSIONS];

    for (dimension, value) in vector.iter_mut().enumerate() {
        *value = i16::from_le(unsafe { *reference.add(dimension) });
    }

    vector
}

struct CostTop {
    indexes: [usize; NEAREST_COUNT],
    distances: [u64; NEAREST_COUNT],
    len: usize,
    farthest_slot: usize,
}

impl CostTop {
    fn new() -> Self {
        Self {
            indexes: [0; NEAREST_COUNT],
            distances: [u64::MAX; NEAREST_COUNT],
            len: 0,
            farthest_slot: 0,
        }
    }

    fn current_worst_distance(&self) -> u64 {
        if self.len < NEAREST_COUNT {
            u64::MAX
        } else {
            self.distances[self.farthest_slot]
        }
    }

    fn add(&mut self, index: usize, distance: u64) {
        if self.len == NEAREST_COUNT && distance >= self.current_worst_distance() {
            return;
        }

        if self.indexes[..self.len]
            .iter()
            .any(|candidate| *candidate == index)
        {
            return;
        }

        if self.len < NEAREST_COUNT {
            self.indexes[self.len] = index;
            self.distances[self.len] = distance;
            self.len += 1;

            if distance > self.distances[self.farthest_slot] {
                self.farthest_slot = self.len - 1;
            }

            return;
        }

        self.indexes[self.farthest_slot] = index;
        self.distances[self.farthest_slot] = distance;
        self.farthest_slot = farthest_slot_in(&self.distances);
    }
}

fn farthest_slot_in<const N: usize>(distances: &[u64; N]) -> usize {
    let mut farthest_slot = 0;

    for slot in 1..distances.len() {
        if distances[slot] > distances[farthest_slot] {
            farthest_slot = slot;
        }
    }

    farthest_slot
}

#[derive(Debug)]
struct IvfLayout {
    reference_count: usize,
    centroid_count: usize,
    centroids_offset: usize,
    offsets_offset: usize,
    indices_offset: usize,
    vectors_offset: usize,
    fraud_offset: usize,
}

impl IvfLayout {
    fn read(bytes: &[u8], expected_reference_count: Option<usize>) -> Result<Self> {
        if bytes.len() < IVF_HEADER_LEN {
            bail!("invalid IVF binary: file is smaller than header");
        }

        if &bytes[..IVF_MAGIC.len()] != IVF_MAGIC {
            bail!("invalid IVF binary: bad magic");
        }

        let count = read_u64_at(bytes, IVF_MAGIC.len()) as usize;

        if let Some(expected_reference_count) = expected_reference_count {
            if count != expected_reference_count {
                bail!(
                    "invalid IVF binary: expected {} references, got {}",
                    expected_reference_count,
                    count
                );
            }
        }

        let centroid_count_offset = IVF_MAGIC.len() + size_of::<u64>();
        let centroid_count = read_u32_at(bytes, centroid_count_offset) as usize;

        if centroid_count == 0 {
            bail!("invalid IVF binary: no centroids");
        }

        let index_count_offset = centroid_count_offset + size_of::<u32>();
        let index_count = read_u64_at(bytes, index_count_offset) as usize;
        let centroids_offset = IVF_HEADER_LEN;
        let offsets_offset = centroids_offset + centroid_count * VECTOR_LEN;
        let indices_offset = offsets_offset + (centroid_count + 1) * size_of::<u32>();
        let vectors_offset = indices_offset + index_count * size_of::<u32>();
        let fraud_offset = vectors_offset + count * VECTOR_LEN;
        let expected_len = fraud_offset + count.div_ceil(8);

        if bytes.len() != expected_len {
            bail!(
                "invalid IVF binary: expected {} bytes, got {} bytes",
                expected_len,
                bytes.len()
            );
        }

        let last_offset =
            read_u32_at(bytes, offsets_offset + centroid_count * size_of::<u32>()) as usize;

        if last_offset != index_count {
            bail!(
                "invalid IVF binary: last offset {} does not match entry count {}",
                last_offset,
                index_count
            );
        }

        Ok(Self {
            reference_count: count,
            centroid_count,
            centroids_offset,
            offsets_offset,
            indices_offset,
            vectors_offset,
            fraud_offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn sample_ivf_bytes(
        reference_count: usize,
        centroid_count: usize,
        entry_count: usize,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(IVF_MAGIC);
        bytes.extend_from_slice(&(reference_count as u64).to_le_bytes());
        bytes.extend_from_slice(&(centroid_count as u32).to_le_bytes());
        bytes.extend_from_slice(&(entry_count as u64).to_le_bytes());

        for _ in 0..centroid_count * VECTOR_DIMENSIONS {
            bytes.extend_from_slice(&0_i16.to_le_bytes());
        }

        for offset in 0..=centroid_count {
            let boundary = offset * entry_count / centroid_count;
            bytes.extend_from_slice(&(boundary as u32).to_le_bytes());
        }

        for position in 0..entry_count {
            let index = match position {
                0 => 2,
                1 => 0,
                2 => 1,
                _ => position as u32,
            };
            bytes.extend_from_slice(&index.to_le_bytes());
        }

        for index in 0..reference_count {
            for dimension in 0..VECTOR_DIMENSIONS {
                let value = if dimension == 0 { index as i16 } else { 0 };
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }

        bytes.resize(bytes.len() + reference_count.div_ceil(8), 0);
        bytes
    }

    fn mmap_bytes(bytes: &[u8]) -> Mmap {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "rinha-ivf-flat-test-{}-{nanos}.bin",
            std::process::id()
        ));
        let mut file = File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        file.flush().unwrap();
        drop(file);

        let file = File::open(&path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };
        let _ = std::fs::remove_file(path);
        mmap
    }

    #[test]
    fn reads_ivf_flat_header_and_offsets() {
        let bytes = sample_ivf_bytes(3, 1, 3);
        let layout = IvfLayout::read(&bytes, Some(3)).unwrap();

        assert_eq!(layout.reference_count, 3);
        assert_eq!(layout.centroid_count, 1);
        assert_eq!(layout.centroids_offset, IVF_HEADER_LEN);
        assert_eq!(
            layout.indices_offset,
            layout.offsets_offset + 2 * size_of::<u32>()
        );
        assert_eq!(
            layout.vectors_offset,
            layout.indices_offset + 3 * size_of::<u32>()
        );
    }

    #[test]
    fn reads_embedded_fraud_labels() {
        let mut bytes = sample_ivf_bytes(3, 1, 3);
        let last = bytes.len() - 1;
        bytes[last] = 0b0000_0101;
        let mmap = mmap_bytes(&bytes);
        let layout = IvfLayout::read(&mmap, Some(3)).unwrap();
        let index = IvfIndex {
            mmap,
            reference_count: layout.reference_count,
            centroid_count: layout.centroid_count,
            centroids_offset: layout.centroids_offset,
            offsets_offset: layout.offsets_offset,
            indices_offset: layout.indices_offset,
            vectors_offset: layout.vectors_offset,
            fraud_offset: layout.fraud_offset,
        };

        assert!(index.is_fraud(0));
        assert!(!index.is_fraud(1));
        assert!(index.is_fraud(2));
    }

    #[test]
    fn rejects_unknown_ivf_magic() {
        let mut bytes = sample_ivf_bytes(3, 1, 3);
        bytes[..IVF_MAGIC.len()].copy_from_slice(b"R26IVF00");

        let error = IvfLayout::read(&bytes, Some(3)).unwrap_err().to_string();

        assert!(error.contains("bad magic"));
    }

    #[test]
    fn rejects_invalid_last_offset() {
        let mut bytes = sample_ivf_bytes(3, 1, 3);
        let offsets_offset = IVF_HEADER_LEN + VECTOR_LEN;
        bytes[offsets_offset + size_of::<u32>()..offsets_offset + 2 * size_of::<u32>()]
            .copy_from_slice(&2_u32.to_le_bytes());

        let error = IvfLayout::read(&bytes, Some(3)).unwrap_err().to_string();

        assert!(error.contains("last offset"));
    }

    #[test]
    fn visits_candidates_with_flat_distances() {
        let bytes = sample_ivf_bytes(3, 1, 3);
        let mmap = mmap_bytes(&bytes);
        let layout = IvfLayout::read(&mmap, Some(3)).unwrap();
        let index = IvfIndex {
            mmap,
            reference_count: layout.reference_count,
            centroid_count: layout.centroid_count,
            centroids_offset: layout.centroids_offset,
            offsets_offset: layout.offsets_offset,
            indices_offset: layout.indices_offset,
            vectors_offset: layout.vectors_offset,
            fraud_offset: layout.fraud_offset,
        };
        let mut visited = Vec::new();

        index.for_each_candidates(
            &[0; VECTOR_DIMENSIONS],
            &mut || u64::MAX,
            &mut |index, distance| visited.push((index, distance)),
        );

        assert_eq!(visited, vec![(2, 4), (0, 0), (1, 1)]);
    }
}
