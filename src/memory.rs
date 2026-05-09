use std::fs::File;
use std::ops::Range;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;

use crate::*;

const IVF_BLOCK_LANES: usize = 8;
const IVF_BLOCK_PAIRS: usize = VECTOR_DIMENSIONS / 2;
const IVF_BLOCK_VECTOR_LEN: usize = IVF_BLOCK_LANES * VECTOR_DIMENSIONS * size_of::<i16>();
const IVF_BLOCK_INDEX_LEN: usize = IVF_BLOCK_LANES * size_of::<u32>();

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceRecord {
    pub vector: ReferenceVector,
    pub is_fraud: bool,
}

pub struct SearchContext {
    coarse_indexes: [usize; IVF_MAX_COARSE_PROBES],
    coarse_len: usize,
    coarse_cost: CentroidSearchCost,
    coarse_centroid_candidates: usize,
}

impl SearchContext {
    pub fn empty() -> Self {
        Self {
            coarse_indexes: [0; IVF_MAX_COARSE_PROBES],
            coarse_len: 0,
            coarse_cost: CentroidSearchCost::default(),
            coarse_centroid_candidates: 0,
        }
    }

    fn from_coarse_selection(selection: CoarseSelection<IVF_MAX_COARSE_PROBES>) -> Self {
        Self {
            coarse_indexes: selection.indexes,
            coarse_len: selection.len,
            coarse_cost: selection.cost,
            coarse_centroid_candidates: selection.coarse_centroid_candidates,
        }
    }

    fn coarse_indexes(&self, limit: usize) -> &[usize] {
        &self.coarse_indexes[..self.coarse_len.min(limit)]
    }
}

pub trait ReferenceSource {
    fn is_fraud(&self, index: usize) -> bool;

    fn prepare_search_context(&self, _vector: &ReferenceVector) -> SearchContext {
        SearchContext::empty()
    }

    fn for_each_primary_candidate_batch<C, V>(
        &self,
        context: &SearchContext,
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
    #[allow(dead_code)]
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

#[allow(dead_code)]
impl SearchCost {
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
    pub boundary_coarse_group_probes: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HierarchySearchCost {
    pub search: SearchCost,
    pub coarse_centroid_candidates: usize,
    pub fine_centroid_candidates: usize,
}

#[allow(dead_code)]
impl HierarchySearchCost {
    pub fn total_units(&self) -> usize {
        self.search.total_units()
    }
}

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

    fn prepare_search_context(&self, vector: &ReferenceVector) -> SearchContext {
        SearchContext::from_coarse_selection(
            self.hierarchy
                .select_coarse_centroids::<{ IVF_MAX_COARSE_PROBES }>(vector),
        )
    }

    fn for_each_primary_candidate_batch<C, V>(
        &self,
        context: &SearchContext,
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
            context,
            vector,
            start_probe..end_probe,
            current_worst_top_distance,
            visit,
        );
    }
}

impl IndexedReferences {
    #[allow(dead_code)]
    pub fn search_cost_for_probe_count(
        &self,
        vector: &ReferenceVector,
        probe_count: usize,
    ) -> HierarchySearchCost {
        let context = self.prepare_search_context(vector);
        self.search_cost_for_probe_count_with_context(&context, vector, probe_count)
    }

    #[allow(dead_code)]
    pub fn search_cost_for_probe_count_with_context(
        &self,
        context: &SearchContext,
        vector: &ReferenceVector,
        probe_count: usize,
    ) -> HierarchySearchCost {
        self.ivfs
            .primary
            .hierarchy_search_cost(&self.hierarchy, context, vector, probe_count)
    }

    #[allow(dead_code)]
    pub fn hierarchy_config(&self) -> HierarchyConfig {
        HierarchyConfig {
            coarse_centroids: IVF_COARSE_CENTROIDS,
            coarse_probes: IVF_COARSE_PROBES,
            coarse_iterations: IVF_COARSE_ITERATIONS,
            boundary_coarse_group_probes: IVF_MAX_COARSE_PROBES,
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
        _context: &SearchContext,
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

        for (index, record) in self.iter().enumerate() {
            let max_useful_distance = current_worst_top_distance();
            let distance = distance2_limited(vector, &record.vector, max_useful_distance);

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
        context: &SearchContext,
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
            context,
            vector,
            start_probe,
            end_probe,
            current_worst_top_distance,
            visit,
        );
    }
}

fn distance2_limited(a: &ReferenceVector, b: &ReferenceVector, limit: u64) -> u64 {
    distance2_vector_limited(a, b, limit).distance
}

fn distance2_vector_limited(a: &ReferenceVector, b: &ReferenceVector, limit: u64) -> DistanceEval {
    let mut distance = distance2_first8_vector(a, b);

    if distance >= limit {
        return DistanceEval {
            distance,
            dimensions: 8,
            early_discard: true,
        };
    }

    let tail_distance = distance2_last8_vector(a, b);
    distance += tail_distance;

    if distance >= limit {
        return DistanceEval {
            distance,
            dimensions: VECTOR_DIMENSIONS,
            early_discard: false,
        };
    }

    DistanceEval {
        distance,
        dimensions: VECTOR_DIMENSIONS,
        early_discard: false,
    }
}

#[inline(always)]
fn distance2_last8_vector(a: &ReferenceVector, b: &ReferenceVector) -> u64 {
    use std::arch::x86_64::*;

    unsafe {
        let left = _mm_loadu_si128(a.as_ptr().add(8).cast::<__m128i>());
        let right = _mm_loadu_si128(b.as_ptr().add(8).cast::<__m128i>());
        distance2_first8_sse2(left, right)
    }
}

#[inline(always)]
fn distance2_first8_vector(a: &ReferenceVector, b: &ReferenceVector) -> u64 {
    use std::arch::x86_64::*;

    unsafe {
        let left = _mm_loadu_si128(a.as_ptr().cast::<__m128i>());
        let right = _mm_loadu_si128(b.as_ptr().cast::<__m128i>());
        distance2_first8_sse2(left, right)
    }
}

#[inline(always)]
fn distance2_first8_mmap(reference: *const i16, vector: &ReferenceVector) -> u64 {
    use std::arch::x86_64::*;

    unsafe {
        let left = _mm_loadu_si128(vector.as_ptr().cast::<__m128i>());
        let right = _mm_loadu_si128(reference.cast::<__m128i>());
        distance2_first8_sse2(left, right)
    }
}

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

#[derive(Clone, Copy, Default)]
struct CentroidSearchCost {
    candidates: usize,
    early_discards: usize,
    full_distance_candidates: usize,
    vector_dimensions_evaluated: usize,
}

#[derive(Clone, Copy)]
enum CentroidCostMode {
    IncludeCoarse,
    FineOnly,
}

impl CentroidCostMode {
    fn initial_cost(self, context: &SearchContext) -> CentroidSearchCost {
        match self {
            Self::IncludeCoarse => context.coarse_cost,
            Self::FineOnly => CentroidSearchCost::default(),
        }
    }

    fn coarse_centroid_candidates(self, context: &SearchContext) -> usize {
        match self {
            Self::IncludeCoarse => context.coarse_centroid_candidates,
            Self::FineOnly => 0,
        }
    }
}

struct CoarseSelection<const N: usize> {
    indexes: [usize; N],
    len: usize,
    cost: CentroidSearchCost,
    coarse_centroid_candidates: usize,
}

struct SelectedCentroids<const N: usize> {
    centroids: [usize; N],
    len: usize,
    #[allow(dead_code)]
    cost: CentroidSearchCost,
    #[allow(dead_code)]
    coarse_centroid_candidates: usize,
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    candidate_count: usize,
    block_count: usize,
    centroids_offset: usize,
    candidate_offsets_offset: usize,
    block_offsets_offset: usize,
    blocks_offset: usize,
    block_indices_offset: usize,
    fraud_offset: usize,
}

struct IvfIndexes {
    primary: IvfIndex,
}

impl CentroidHierarchy {
    fn select_coarse_centroids<const N: usize>(
        &self,
        vector: &ReferenceVector,
    ) -> CoarseSelection<N> {
        let mut coarse_nearest = TopCentroids::<N>::new();
        let mut cost = CentroidSearchCost::default();
        let mut coarse_centroid_candidates = 0;

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

        coarse_nearest.sort_indexes();

        CoarseSelection {
            indexes: coarse_nearest.indexes,
            len: coarse_nearest.len,
            cost,
            coarse_centroid_candidates,
        }
    }

    fn select_top_fine_centroids_from_context_coarse<const FINE_PROBES: usize>(
        &self,
        index: &IvfIndex,
        vector: &ReferenceVector,
        context: &SearchContext,
        coarse_limit: usize,
        cost_mode: CentroidCostMode,
    ) -> SelectedCentroids<FINE_PROBES> {
        let mut fine_nearest = TopCentroids::<FINE_PROBES>::new();
        let mut cost = cost_mode.initial_cost(context);
        let coarse_centroid_candidates = cost_mode.coarse_centroid_candidates(context);
        let mut fine_centroid_candidates = 0;

        for coarse in context.coarse_indexes(coarse_limit).iter().copied() {
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
            primary: IvfIndex::load("resources/ivf.bin", None)?,
        })
    }

    fn is_fraud(&self, index: usize) -> bool {
        self.primary.is_fraud(index)
    }
}

fn apply_centroid_search_cost(search: &mut SearchCost, cost: CentroidSearchCost) {
    search.centroid_candidates = cost.candidates;
    search.centroid_early_discards = cost.early_discards;
    search.centroid_full_distance_candidates = cost.full_distance_candidates;
    search.centroid_vector_dimensions_evaluated = cost.vector_dimensions_evaluated;
}

impl IvfIndex {
    fn load(path: &str, reference_count: Option<usize>) -> Result<Self> {
        let input_file = File::open(path)
            .with_context(|| format!("failed to open required IVF index at {path}"))?;
        let mmap = unsafe { Mmap::map(&input_file) }?;
        let layout = IvfLayout::read(&mmap, reference_count)?;

        Ok(Self {
            mmap,
            reference_count: layout.reference_count,
            centroid_count: layout.centroid_count,
            candidate_count: layout.candidate_count,
            block_count: layout.block_count,
            centroids_offset: layout.centroids_offset,
            candidate_offsets_offset: layout.candidate_offsets_offset,
            block_offsets_offset: layout.block_offsets_offset,
            blocks_offset: layout.blocks_offset,
            block_indices_offset: layout.block_indices_offset,
            fraud_offset: layout.fraud_offset,
        })
    }

    fn is_fraud(&self, index: usize) -> bool {
        debug_assert!(index < self.reference_count);
        let byte = self.mmap[self.fraud_offset + index / 8];
        byte & (1 << (index % 8)) != 0
    }

    fn for_each_hierarchy_candidate_batch<C, V>(
        &self,
        hierarchy: &CentroidHierarchy,
        context: &SearchContext,
        vector: &ReferenceVector,
        probe_range: Range<usize>,
        current_worst_top_distance: &mut C,
        visit: &mut V,
    ) where
        C: FnMut() -> u64,
        V: FnMut(usize, u64),
    {
        self.for_each_selected_hierarchy_candidate_batch::<IVF_FINE_PROBES, _, _>(
            hierarchy,
            context,
            vector,
            probe_range,
            current_worst_top_distance,
            visit,
        );
    }

    fn for_each_selected_hierarchy_candidate_batch<const N: usize, C, V>(
        &self,
        hierarchy: &CentroidHierarchy,
        context: &SearchContext,
        vector: &ReferenceVector,
        probe_range: Range<usize>,
        current_worst_top_distance: &mut C,
        visit: &mut V,
    ) where
        C: FnMut() -> u64,
        V: FnMut(usize, u64),
    {
        let selection = hierarchy.select_top_fine_centroids_from_context_coarse::<N>(
            self,
            vector,
            context,
            IVF_COARSE_PROBES,
            CentroidCostMode::FineOnly,
        );
        let start = probe_range.start.min(selection.len);
        let end = probe_range.end.min(selection.len);

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
        let use_avx2 = is_x86_feature_detected!("avx2");
        let query_pairs = if use_avx2 {
            Some(unsafe { candidate_query_pairs_avx2(vector) })
        } else {
            None
        };

        for centroid in centroids.iter().copied() {
            let candidate_start = self.candidate_offset_at(centroid) as usize;
            let candidate_end = self.candidate_offset_at(centroid + 1) as usize;
            let block_start = self.block_offset_at(centroid) as usize;
            let block_end = self.block_offset_at(centroid + 1) as usize;

            for (local_block_index, block) in (block_start..block_end).enumerate() {
                let remaining =
                    candidate_end - (candidate_start + local_block_index * IVF_BLOCK_LANES);
                let valid_lanes = remaining.min(IVF_BLOCK_LANES);
                let max_useful_distance = current_worst_top_distance();
                let block_ptr = self.block_vector_ptr(block);
                let distances = if let Some(query_pairs) = &query_pairs {
                    unsafe {
                        candidate_block_distances_avx2_limited(
                            block_ptr,
                            query_pairs,
                            valid_lanes,
                            max_useful_distance,
                        )
                    }
                } else {
                    candidate_block_distances_scalar(
                        block_ptr,
                        vector,
                        valid_lanes,
                        max_useful_distance,
                    )
                };

                for (lane, distance) in distances.iter().copied().enumerate().take(valid_lanes) {
                    if !is_candidate_distance_useful(distance, max_useful_distance) {
                        continue;
                    }

                    let index = self.block_index_at(block, lane);
                    if index == u32::MAX {
                        continue;
                    }

                    visit(index as usize, distance);
                }
            }
        }
    }

    #[allow(dead_code)]
    fn hierarchy_search_cost(
        &self,
        hierarchy: &CentroidHierarchy,
        context: &SearchContext,
        vector: &ReferenceVector,
        probe_count: usize,
    ) -> HierarchySearchCost {
        self.hierarchy_search_cost_for::<IVF_FINE_PROBES>(hierarchy, context, vector, probe_count)
    }

    #[allow(dead_code)]
    fn hierarchy_search_cost_for<const N: usize>(
        &self,
        hierarchy: &CentroidHierarchy,
        context: &SearchContext,
        vector: &ReferenceVector,
        probe_count: usize,
    ) -> HierarchySearchCost {
        let selection = hierarchy.select_top_fine_centroids_from_context_coarse::<N>(
            self,
            vector,
            context,
            IVF_COARSE_PROBES,
            CentroidCostMode::IncludeCoarse,
        );
        let selected_len = probe_count.min(selection.len);
        self.search_cost_from_selected_centroids(&selection, selected_len, vector)
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
                let start = self.candidate_offset_at(*centroid) as usize;
                let end = self.candidate_offset_at(*centroid + 1) as usize;
                end - start
            })
            .sum();
        let mut nearest = CostTop::new();
        let mut cost = SearchCost {
            primary_list_candidates,
            ..SearchCost::default()
        };

        for centroid in centroids.iter().copied() {
            let candidate_start = self.candidate_offset_at(centroid) as usize;
            let candidate_end = self.candidate_offset_at(centroid + 1) as usize;
            let block_start = self.block_offset_at(centroid) as usize;
            let block_end = self.block_offset_at(centroid + 1) as usize;

            for (local_block_index, block) in (block_start..block_end).enumerate() {
                let remaining =
                    candidate_end - (candidate_start + local_block_index * IVF_BLOCK_LANES);
                let valid_lanes = remaining.min(IVF_BLOCK_LANES);
                let limit = nearest.current_worst_distance();
                let evaluation = candidate_block_distances_scalar_eval(
                    self.block_vector_ptr(block),
                    vector,
                    valid_lanes,
                    limit,
                );

                cost.flat_candidates += valid_lanes;
                cost.flat_vector_dimensions_evaluated += evaluation.dimensions * valid_lanes;

                if evaluation.early_discard {
                    cost.flat_early_discards += valid_lanes;
                    continue;
                } else {
                    cost.flat_full_distance_candidates += valid_lanes;
                }

                for (lane, distance) in evaluation
                    .distances
                    .iter()
                    .copied()
                    .enumerate()
                    .take(valid_lanes)
                {
                    let index = self.block_index_at(block, lane);
                    if index == u32::MAX {
                        continue;
                    }

                    if is_candidate_distance_useful(distance, limit) {
                        nearest.add(index as usize, distance);
                    }
                }
            }
        }

        cost
    }

    #[allow(dead_code)]
    fn search_cost_from_selected_centroids<const N: usize>(
        &self,
        selection: &SelectedCentroids<N>,
        selected_len: usize,
        vector: &ReferenceVector,
    ) -> HierarchySearchCost {
        let mut search =
            self.search_cost_from_centroids(&selection.centroids[..selected_len], vector);
        apply_centroid_search_cost(&mut search, selection.cost);

        HierarchySearchCost {
            search,
            coarse_centroid_candidates: selection.coarse_centroid_candidates,
            fine_centroid_candidates: selection.fine_centroid_candidates,
        }
    }

    fn build_centroid_hierarchy(&self) -> CentroidHierarchy {
        let fine_centroid_vectors = (0..self.centroid_count)
            .map(|centroid| self.centroid_vector_at(centroid))
            .collect::<Vec<_>>();
        let coarse_count = IVF_COARSE_CENTROIDS.min(fine_centroid_vectors.len()).max(1);
        let mut coarse_centroids = initial_sampled_centroids(&fine_centroid_vectors, coarse_count);

        for _ in 0..IVF_COARSE_ITERATIONS {
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

    fn candidate_offset_at(&self, centroid: usize) -> u32 {
        read_u32_at(&self.mmap, self.candidate_offset_offset(centroid))
    }

    fn block_offset_at(&self, centroid: usize) -> u32 {
        read_u32_at(&self.mmap, self.block_offset_offset(centroid))
    }

    fn block_index_at(&self, block: usize, lane: usize) -> u32 {
        debug_assert!(block < self.block_count);
        debug_assert!(lane < IVF_BLOCK_LANES);
        read_u32_at(&self.mmap, self.block_index_offset(block, lane))
    }

    fn centroid_vector_at(&self, centroid: usize) -> ReferenceVector {
        vector_mmap_at(&self.mmap, self.centroid_vector_offset(centroid))
    }

    fn centroid_vector_offset(&self, centroid: usize) -> usize {
        self.centroids_offset + centroid * VECTOR_LEN
    }

    fn candidate_offset_offset(&self, centroid: usize) -> usize {
        self.candidate_offsets_offset + centroid * size_of::<u32>()
    }

    fn block_offset_offset(&self, centroid: usize) -> usize {
        self.block_offsets_offset + centroid * size_of::<u32>()
    }

    fn block_vector_offset(&self, block: usize) -> usize {
        debug_assert!(block < self.block_count);
        self.blocks_offset + block * IVF_BLOCK_VECTOR_LEN
    }

    fn block_vector_ptr(&self, block: usize) -> *const i16 {
        unsafe {
            self.mmap
                .as_ptr()
                .add(self.block_vector_offset(block))
                .cast::<i16>()
        }
    }

    fn block_index_offset(&self, block: usize, lane: usize) -> usize {
        self.block_indices_offset + (block * IVF_BLOCK_LANES + lane) * size_of::<u32>()
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
            dimensions: 8,
            early_discard: true,
        };
    }

    distance += distance2_last8_mmap(reference, vector);

    if distance >= limit {
        return DistanceEval {
            distance,
            dimensions: VECTOR_DIMENSIONS,
            early_discard: false,
        };
    }

    DistanceEval {
        distance,
        dimensions: VECTOR_DIMENSIONS,
        early_discard: false,
    }
}

#[inline(always)]
fn distance2_last8_mmap(reference: *const i16, vector: &ReferenceVector) -> u64 {
    use std::arch::x86_64::*;

    unsafe {
        let left = _mm_loadu_si128(vector.as_ptr().add(8).cast::<__m128i>());
        let right = _mm_loadu_si128(reference.add(8).cast::<__m128i>());
        distance2_first8_sse2(left, right)
    }
}

#[target_feature(enable = "avx2")]
unsafe fn candidate_query_pairs_avx2(
    vector: &ReferenceVector,
) -> [std::arch::x86_64::__m256i; IVF_BLOCK_PAIRS] {
    use std::arch::x86_64::*;

    let mut pairs = [_mm256_setzero_si256(); IVF_BLOCK_PAIRS];

    for pair in 0..IVF_BLOCK_PAIRS {
        let first = vector[pair * 2];
        let second = vector[pair * 2 + 1];
        pairs[pair] = _mm256_setr_epi16(
            first, second, first, second, first, second, first, second, first, second, first,
            second, first, second, first, second,
        );
    }

    pairs
}

#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn candidate_block_distances_avx2(
    block_ptr: *const i16,
    query_pairs: &[std::arch::x86_64::__m256i; IVF_BLOCK_PAIRS],
) -> [u64; IVF_BLOCK_LANES] {
    unsafe {
        candidate_block_distances_avx2_limited(block_ptr, query_pairs, IVF_BLOCK_LANES, u64::MAX)
    }
}

#[target_feature(enable = "avx2")]
unsafe fn candidate_block_distances_avx2_limited(
    block_ptr: *const i16,
    query_pairs: &[std::arch::x86_64::__m256i; IVF_BLOCK_PAIRS],
    valid_lanes: usize,
    limit: u64,
) -> [u64; IVF_BLOCK_LANES] {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_si256();

    for (pair, query_pair) in query_pairs.iter().enumerate().take(4) {
        let refs = unsafe { _mm256_loadu_si256(block_ptr.add(pair * 16).cast::<__m256i>()) };
        let diff = _mm256_sub_epi16(refs, *query_pair);
        let partial = _mm256_madd_epi16(diff, diff);
        acc = _mm256_add_epi32(acc, partial);
    }

    let mut lanes = [0_i32; IVF_BLOCK_LANES];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), acc) };

    if valid_lanes > 0
        && lanes[..valid_lanes]
            .iter()
            .all(|partial| *partial as u64 >= limit)
    {
        return [u64::MAX; IVF_BLOCK_LANES];
    }

    for (pair, query_pair) in query_pairs.iter().enumerate().skip(4) {
        let refs = unsafe { _mm256_loadu_si256(block_ptr.add(pair * 16).cast::<__m256i>()) };
        let diff = _mm256_sub_epi16(refs, *query_pair);
        let partial = _mm256_madd_epi16(diff, diff);
        acc = _mm256_add_epi32(acc, partial);
    }

    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), acc) };

    let mut distances = [u64::MAX; IVF_BLOCK_LANES];
    for lane in 0..valid_lanes {
        distances[lane] = lanes[lane] as u64;
    }

    distances
}

fn candidate_block_distances_scalar(
    block_ptr: *const i16,
    vector: &ReferenceVector,
    valid_lanes: usize,
    limit: u64,
) -> [u64; IVF_BLOCK_LANES] {
    candidate_block_distances_scalar_eval(block_ptr, vector, valid_lanes, limit).distances
}

fn candidate_block_distances_scalar_eval(
    block_ptr: *const i16,
    vector: &ReferenceVector,
    valid_lanes: usize,
    limit: u64,
) -> CandidateBlockEval {
    let mut distances = [u64::MAX; IVF_BLOCK_LANES];

    for lane in 0..valid_lanes {
        let mut distance = 0_u64;

        for pair in 0..4 {
            let first = unsafe { *block_ptr.add(pair * 16 + lane * 2) };
            let second = unsafe { *block_ptr.add(pair * 16 + lane * 2 + 1) };
            let first_delta = i32::from(first) - i32::from(vector[pair * 2]);
            let second_delta = i32::from(second) - i32::from(vector[pair * 2 + 1]);
            distance += (first_delta * first_delta + second_delta * second_delta) as u64;
        }

        distances[lane] = distance;
    }

    if valid_lanes > 0
        && distances[..valid_lanes]
            .iter()
            .all(|distance| *distance >= limit)
    {
        return CandidateBlockEval {
            distances,
            dimensions: 8,
            early_discard: true,
        };
    }

    for lane in 0..valid_lanes {
        let mut distance = distances[lane];

        for pair in 4..IVF_BLOCK_PAIRS {
            let first = unsafe { *block_ptr.add(pair * 16 + lane * 2) };
            let second = unsafe { *block_ptr.add(pair * 16 + lane * 2 + 1) };
            let first_delta = i32::from(first) - i32::from(vector[pair * 2]);
            let second_delta = i32::from(second) - i32::from(vector[pair * 2 + 1]);
            distance += (first_delta * first_delta + second_delta * second_delta) as u64;
        }

        distances[lane] = distance;
    }

    CandidateBlockEval {
        distances,
        dimensions: VECTOR_DIMENSIONS,
        early_discard: false,
    }
}

struct CandidateBlockEval {
    distances: [u64; IVF_BLOCK_LANES],
    dimensions: usize,
    early_discard: bool,
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

        if self.indexes[..self.len].contains(&index) {
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
    candidate_count: usize,
    block_count: usize,
    centroids_offset: usize,
    candidate_offsets_offset: usize,
    block_offsets_offset: usize,
    blocks_offset: usize,
    block_indices_offset: usize,
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

        if let Some(expected_reference_count) = expected_reference_count
            && count != expected_reference_count
        {
            bail!(
                "invalid IVF binary: expected {} references, got {}",
                expected_reference_count,
                count
            );
        }

        let centroid_count_offset = IVF_MAGIC.len() + size_of::<u64>();
        let centroid_count = read_u32_at(bytes, centroid_count_offset) as usize;

        if centroid_count == 0 {
            bail!("invalid IVF binary: no centroids");
        }

        let candidate_count_offset = centroid_count_offset + size_of::<u32>();
        let candidate_count = read_u64_at(bytes, candidate_count_offset) as usize;
        let block_count_offset = candidate_count_offset + size_of::<u64>();
        let block_count_u64 = read_u64_at(bytes, block_count_offset);

        if block_count_u64 > u64::from(u32::MAX) {
            bail!("invalid IVF binary: block count exceeds u32::MAX");
        }

        let block_count = block_count_u64 as usize;
        let centroids_offset = IVF_HEADER_LEN;
        let candidate_offsets_offset = centroids_offset + centroid_count * VECTOR_LEN;
        let block_offsets_offset =
            candidate_offsets_offset + (centroid_count + 1) * size_of::<u32>();
        let blocks_offset = block_offsets_offset + (centroid_count + 1) * size_of::<u32>();
        let block_indices_offset = blocks_offset + block_count * IVF_BLOCK_VECTOR_LEN;
        let fraud_offset = block_indices_offset + block_count * IVF_BLOCK_INDEX_LEN;
        let expected_len = fraud_offset + count.div_ceil(8);

        if bytes.len() != expected_len {
            bail!(
                "invalid IVF binary: expected {} bytes, got {} bytes",
                expected_len,
                bytes.len()
            );
        }

        let last_candidate_offset = read_u32_at(
            bytes,
            candidate_offsets_offset + centroid_count * size_of::<u32>(),
        ) as usize;

        if last_candidate_offset != candidate_count {
            bail!(
                "invalid IVF binary: last candidate offset {} does not match candidate count {}",
                last_candidate_offset,
                candidate_count
            );
        }

        let last_block_offset = read_u32_at(
            bytes,
            block_offsets_offset + centroid_count * size_of::<u32>(),
        ) as usize;

        if last_block_offset != block_count {
            bail!(
                "invalid IVF binary: last block offset {} does not match block count {}",
                last_block_offset,
                block_count
            );
        }

        Ok(Self {
            reference_count: count,
            centroid_count,
            candidate_count,
            block_count,
            centroids_offset,
            candidate_offsets_offset,
            block_offsets_offset,
            blocks_offset,
            block_indices_offset,
            fraud_offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn sample_ivf_bytes(reference_count: usize, candidate_lists: &[Vec<u32>]) -> Vec<u8> {
        let centroid_count = candidate_lists.len();
        let candidate_count = candidate_lists.iter().map(Vec::len).sum::<usize>();
        let mut candidate_offsets = vec![0_u32; centroid_count + 1];
        let mut block_offsets = vec![0_u32; centroid_count + 1];

        for centroid in 0..centroid_count {
            candidate_offsets[centroid + 1] =
                candidate_offsets[centroid] + candidate_lists[centroid].len() as u32;
            block_offsets[centroid + 1] =
                block_offsets[centroid] + (candidate_lists[centroid].len() as u32).div_ceil(8);
        }

        let block_count = *block_offsets.last().unwrap() as usize;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(IVF_MAGIC);
        bytes.extend_from_slice(&(reference_count as u64).to_le_bytes());
        bytes.extend_from_slice(&(centroid_count as u32).to_le_bytes());
        bytes.extend_from_slice(&(candidate_count as u64).to_le_bytes());
        bytes.extend_from_slice(&(block_count as u64).to_le_bytes());

        for _ in 0..centroid_count * VECTOR_DIMENSIONS {
            bytes.extend_from_slice(&0_i16.to_le_bytes());
        }

        for offset in &candidate_offsets {
            bytes.extend_from_slice(&offset.to_le_bytes());
        }

        for offset in &block_offsets {
            bytes.extend_from_slice(&offset.to_le_bytes());
        }

        for list in candidate_lists {
            for chunk in list.chunks(8) {
                let mut lanes = [[0_i16; VECTOR_DIMENSIONS]; 8];
                for (lane, index) in chunk.iter().copied().enumerate() {
                    lanes[lane][0] = index as i16;
                }

                append_block_vectors(&mut bytes, &lanes);
            }
        }

        for list in candidate_lists {
            for chunk in list.chunks(8) {
                for lane in 0..8 {
                    let index = chunk.get(lane).copied().unwrap_or(u32::MAX);
                    bytes.extend_from_slice(&index.to_le_bytes());
                }
            }
        }

        bytes.resize(bytes.len() + reference_count.div_ceil(8), 0);
        bytes
    }

    fn append_block_vectors(bytes: &mut Vec<u8>, lanes: &[[i16; VECTOR_DIMENSIONS]; 8]) {
        for pair in 0..VECTOR_DIMENSIONS / 2 {
            for lane in lanes {
                bytes.extend_from_slice(&lane[pair * 2].to_le_bytes());
                bytes.extend_from_slice(&lane[pair * 2 + 1].to_le_bytes());
            }
        }
    }

    fn sample_index(bytes: Vec<u8>, reference_count: usize) -> IvfIndex {
        let mmap = mmap_bytes(&bytes);
        let layout = IvfLayout::read(&mmap, Some(reference_count)).unwrap();
        IvfIndex {
            mmap,
            reference_count: layout.reference_count,
            centroid_count: layout.centroid_count,
            candidate_count: layout.candidate_count,
            block_count: layout.block_count,
            centroids_offset: layout.centroids_offset,
            candidate_offsets_offset: layout.candidate_offsets_offset,
            block_offsets_offset: layout.block_offsets_offset,
            blocks_offset: layout.blocks_offset,
            block_indices_offset: layout.block_indices_offset,
            fraud_offset: layout.fraud_offset,
        }
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
    fn reads_ivf_blocked_header_and_offsets() {
        let bytes = sample_ivf_bytes(0, &[vec![]]);
        let layout = IvfLayout::read(&bytes, Some(0)).unwrap();

        assert_eq!(layout.reference_count, 0);
        assert_eq!(layout.centroid_count, 1);
        assert_eq!(layout.candidate_count, 0);
        assert_eq!(layout.block_count, 0);
        assert_eq!(layout.centroids_offset, IVF_HEADER_LEN);
        assert_eq!(
            layout.block_offsets_offset,
            layout.candidate_offsets_offset + 2 * size_of::<u32>()
        );
        assert_eq!(
            layout.blocks_offset,
            layout.block_offsets_offset + 2 * size_of::<u32>()
        );
    }

    #[test]
    fn reads_embedded_fraud_labels() {
        let mut bytes = sample_ivf_bytes(3, &[vec![2, 0, 1]]);
        let last = bytes.len() - 1;
        bytes[last] = 0b0000_0101;
        let index = sample_index(bytes, 3);

        assert!(index.is_fraud(0));
        assert!(!index.is_fraud(1));
        assert!(index.is_fraud(2));
    }

    #[test]
    fn rejects_unknown_ivf_magic() {
        let mut bytes = sample_ivf_bytes(3, &[vec![2, 0, 1]]);
        bytes[..IVF_MAGIC.len()].copy_from_slice(b"R26IVF00");

        let error = IvfLayout::read(&bytes, Some(3)).unwrap_err().to_string();

        assert!(error.contains("bad magic"));
    }

    #[test]
    fn rejects_old_ivf_magic() {
        let mut bytes = sample_ivf_bytes(3, &[vec![2, 0, 1]]);
        bytes[..IVF_MAGIC.len()].copy_from_slice(b"R26IFL04");

        let error = IvfLayout::read(&bytes, Some(3)).unwrap_err().to_string();

        assert!(error.contains("bad magic"));
    }

    #[test]
    fn rejects_invalid_size() {
        let mut bytes = sample_ivf_bytes(3, &[vec![2, 0, 1]]);
        bytes.pop();

        let error = IvfLayout::read(&bytes, Some(3)).unwrap_err().to_string();

        assert!(error.contains("expected"));
    }

    #[test]
    fn rejects_invalid_last_candidate_offset() {
        let mut bytes = sample_ivf_bytes(3, &[vec![2, 0, 1]]);
        let offsets_offset = IVF_HEADER_LEN + VECTOR_LEN;
        bytes[offsets_offset + size_of::<u32>()..offsets_offset + 2 * size_of::<u32>()]
            .copy_from_slice(&2_u32.to_le_bytes());

        let error = IvfLayout::read(&bytes, Some(3)).unwrap_err().to_string();

        assert!(error.contains("last candidate offset"));
    }

    #[test]
    fn rejects_invalid_last_block_offset() {
        let mut bytes = sample_ivf_bytes(3, &[vec![2, 0, 1]]);
        let block_offsets_offset = IVF_HEADER_LEN + VECTOR_LEN + 2 * size_of::<u32>();
        bytes[block_offsets_offset + size_of::<u32>()..block_offsets_offset + 2 * size_of::<u32>()]
            .copy_from_slice(&2_u32.to_le_bytes());

        let error = IvfLayout::read(&bytes, Some(3)).unwrap_err().to_string();

        assert!(error.contains("last block offset"));
    }

    #[test]
    fn visits_candidates_with_block_distances() {
        let bytes = sample_ivf_bytes(3, &[vec![2, 0, 1]]);
        let index = sample_index(bytes, 3);
        let mut visited = Vec::new();

        index.for_each_candidates_from_centroids(
            &[0],
            &[0; VECTOR_DIMENSIONS],
            &mut || u64::MAX,
            &mut |index, distance| visited.push((index, distance)),
        );

        assert_eq!(visited, vec![(2, 4), (0, 0), (1, 1)]);
    }

    #[test]
    fn ignores_invalid_lanes_in_last_partial_block() {
        let bytes = sample_ivf_bytes(9, &[vec![0, 1, 2, 3, 4, 5, 6, 7, 8]]);
        let index = sample_index(bytes, 9);
        let mut visited = Vec::new();

        index.for_each_candidates_from_centroids(
            &[0],
            &[0; VECTOR_DIMENSIONS],
            &mut || u64::MAX,
            &mut |index, distance| visited.push((index, distance)),
        );

        assert_eq!(visited.len(), 9);
        assert_eq!(visited[8], (8, 64));
    }

    #[test]
    fn scalar_and_avx2_block_distances_match() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let cases = [
            ([0_i16; VECTOR_DIMENSIONS], [0_i16; VECTOR_DIMENSIONS], 8),
            (
                [0, 0, 0, 0, 0, -10_000, -10_000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0_i16; VECTOR_DIMENSIONS],
                8,
            ),
            (
                [10_000_i16; VECTOR_DIMENSIONS],
                [0_i16; VECTOR_DIMENSIONS],
                8,
            ),
            ([3_i16; VECTOR_DIMENSIONS], [1_i16; VECTOR_DIMENSIONS], 5),
        ];

        for (lane_vector, query, valid_lanes) in cases {
            let mut lanes = [[0_i16; VECTOR_DIMENSIONS]; 8];
            lanes.fill(lane_vector);
            let mut block = Vec::new();
            append_block_vectors(&mut block, &lanes);
            let block_ptr = block.as_ptr().cast::<i16>();
            let scalar = candidate_block_distances_scalar(block_ptr, &query, valid_lanes, u64::MAX);
            let query_pairs = unsafe { candidate_query_pairs_avx2(&query) };
            let avx2 = unsafe {
                if valid_lanes == 8 {
                    candidate_block_distances_avx2(block_ptr, &query_pairs)
                } else {
                    candidate_block_distances_avx2_limited(
                        block_ptr,
                        &query_pairs,
                        valid_lanes,
                        u64::MAX,
                    )
                }
            };

            assert_eq!(avx2, scalar);
        }
    }
}
