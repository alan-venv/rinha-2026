use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;

use crate::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceRecord {
    pub vector: ReferenceVector,
    pub is_fraud: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NearestCandidate {
    pub index: usize,
    pub distance: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct HnswInfo {
    pub m: usize,
    pub upper_m: usize,
    pub ef_search: usize,
    pub entrypoint: usize,
    pub max_level: u8,
    pub reference_count: usize,
}

pub trait ReferenceSource {
    fn is_fraud(&self, index: usize) -> bool;
    fn search_top(&self, vector: &ReferenceVector, top: usize) -> Vec<NearestCandidate>;
}

pub struct IndexedReferences {
    index: HnswIndex,
}

pub fn load_references() -> Result<IndexedReferences> {
    Ok(IndexedReferences {
        index: HnswIndex::load("resources/hnsw.bin")?,
    })
}

impl IndexedReferences {
    pub fn hnsw_info(&self) -> HnswInfo {
        self.index.info()
    }
}

impl ReferenceSource for IndexedReferences {
    fn is_fraud(&self, index: usize) -> bool {
        self.index.is_fraud(index)
    }

    fn search_top(&self, vector: &ReferenceVector, top: usize) -> Vec<NearestCandidate> {
        self.index.search_top(vector, top)
    }
}

impl ReferenceSource for [ReferenceRecord] {
    fn is_fraud(&self, index: usize) -> bool {
        self[index].is_fraud
    }

    fn search_top(&self, vector: &ReferenceVector, top: usize) -> Vec<NearestCandidate> {
        let mut nearest = TopNearest::new(top);

        for (index, record) in self.iter().enumerate() {
            let limit = nearest.current_worst_distance();
            let distance = distance2_limited(vector, &record.vector, limit);

            if limit == u64::MAX || distance < limit {
                nearest.add(NearestCandidate { index, distance });
            }
        }

        nearest.into_sorted()
    }
}

impl<const N: usize> ReferenceSource for [ReferenceRecord; N] {
    fn is_fraud(&self, index: usize) -> bool {
        self[index].is_fraud
    }

    fn search_top(&self, vector: &ReferenceVector, top: usize) -> Vec<NearestCandidate> {
        self.as_slice().search_top(vector, top)
    }
}

struct HnswIndex {
    mmap: Mmap,
    reference_count: usize,
    entrypoint: usize,
    max_level: u8,
    vectors_offset: usize,
    fraud_offset: usize,
    levels_offset: usize,
    level0_offset: usize,
    upper_offsets_offset: usize,
    upper_neighbors_offset: usize,
}

impl HnswIndex {
    fn load(path: &str) -> Result<Self> {
        let input_file = File::open(path)
            .with_context(|| format!("failed to open required HNSW index at {path}"))?;
        let mmap = unsafe { Mmap::map(&input_file) }?;
        let layout = HnswLayout::read(&mmap)?;

        Ok(Self {
            mmap,
            reference_count: layout.reference_count,
            entrypoint: layout.entrypoint,
            max_level: layout.max_level,
            vectors_offset: layout.vectors_offset,
            fraud_offset: layout.fraud_offset,
            levels_offset: layout.levels_offset,
            level0_offset: layout.level0_offset,
            upper_offsets_offset: layout.upper_offsets_offset,
            upper_neighbors_offset: layout.upper_neighbors_offset,
        })
    }

    fn info(&self) -> HnswInfo {
        HnswInfo {
            m: HNSW_M,
            upper_m: HNSW_UPPER_M,
            ef_search: HNSW_EF_SEARCH,
            entrypoint: self.entrypoint,
            max_level: self.max_level,
            reference_count: self.reference_count,
        }
    }

    fn is_fraud(&self, index: usize) -> bool {
        debug_assert!(index < self.reference_count);
        let byte = self.mmap[self.fraud_offset + index / 8];
        byte & (1 << (index % 8)) != 0
    }

    fn search_top(&self, vector: &ReferenceVector, top: usize) -> Vec<NearestCandidate> {
        if self.reference_count == 0 || top == 0 {
            return Vec::new();
        }

        let mut entrypoint = self.entrypoint;
        let mut entry_distance = self.distance_to(entrypoint, vector, u64::MAX);

        for level in (2..=self.max_level).rev() {
            let mut changed = true;

            while changed {
                changed = false;

                for neighbor in self.upper_neighbors(entrypoint, level) {
                    let distance = self.distance_to(neighbor, vector, entry_distance);
                    if distance < entry_distance {
                        entrypoint = neighbor;
                        entry_distance = distance;
                        changed = true;
                    }
                }
            }
        }

        let entrypoints = if self.max_level >= 1 {
            self.search_upper_layer(vector, entrypoint, entry_distance, 1, HNSW_ENTRYPOINTS)
        } else {
            vec![NearestCandidate {
                index: entrypoint,
                distance: entry_distance,
            }]
        };

        self.search_layer0(vector, &entrypoints, top)
    }

    fn search_upper_layer(
        &self,
        vector: &ReferenceVector,
        entrypoint: usize,
        entry_distance: u64,
        level: u8,
        ef: usize,
    ) -> Vec<NearestCandidate> {
        let mut visited = Vec::with_capacity(ef * HNSW_UPPER_M);
        let mut candidates = BinaryHeap::new();
        let mut nearest = TopNearest::new(ef);

        visited.push(entrypoint);
        candidates.push(Reverse(HeapCandidate {
            distance: entry_distance,
            index: entrypoint,
        }));
        nearest.add(NearestCandidate {
            index: entrypoint,
            distance: entry_distance,
        });

        while let Some(Reverse(candidate)) = candidates.pop() {
            if nearest.is_full() && candidate.distance > nearest.current_worst_distance() {
                break;
            }

            for neighbor in self.upper_neighbors(candidate.index, level) {
                if visited.contains(&neighbor) {
                    continue;
                }
                visited.push(neighbor);

                let limit = nearest.current_worst_distance();
                let distance = self.distance_to(neighbor, vector, limit);

                if limit == u64::MAX || distance < limit {
                    candidates.push(Reverse(HeapCandidate {
                        distance,
                        index: neighbor,
                    }));
                    nearest.add(NearestCandidate {
                        index: neighbor,
                        distance,
                    });
                }
            }
        }

        nearest.into_sorted()
    }

    fn search_layer0(
        &self,
        vector: &ReferenceVector,
        entrypoints: &[NearestCandidate],
        top: usize,
    ) -> Vec<NearestCandidate> {
        let mut visited = Vec::with_capacity(HNSW_EF_SEARCH * HNSW_M);
        let mut candidates = BinaryHeap::new();
        let mut nearest = TopNearest::new(HNSW_EF_SEARCH.max(top));

        for entrypoint in entrypoints {
            if visited.contains(&entrypoint.index) {
                continue;
            }

            visited.push(entrypoint.index);
            candidates.push(Reverse(HeapCandidate {
                distance: entrypoint.distance,
                index: entrypoint.index,
            }));
            nearest.add(*entrypoint);
        }

        while let Some(Reverse(candidate)) = candidates.pop() {
            if nearest.is_full() && candidate.distance > nearest.current_worst_distance() {
                break;
            }

            for neighbor in self.level0_neighbors(candidate.index) {
                if visited.contains(&neighbor) {
                    continue;
                }
                visited.push(neighbor);

                let limit = nearest.current_worst_distance();
                let distance = self.distance_to(neighbor, vector, limit);

                if limit == u64::MAX || distance < limit {
                    let found = NearestCandidate {
                        index: neighbor,
                        distance,
                    };
                    candidates.push(Reverse(HeapCandidate {
                        distance,
                        index: neighbor,
                    }));
                    nearest.add(found);
                }
            }
        }

        let mut result = nearest.into_sorted();
        result.truncate(top);
        result
    }

    fn distance_to(&self, index: usize, vector: &ReferenceVector, limit: u64) -> u64 {
        distance2_ptr_limited(self.reference_vector_ptr(index), vector, limit)
    }

    fn level0_neighbors(&self, index: usize) -> impl Iterator<Item = usize> + '_ {
        (0..HNSW_M).filter_map(move |slot| {
            let neighbor = read_u32_at(&self.mmap, self.level0_neighbor_offset(index, slot));
            (neighbor != EMPTY_NEIGHBOR).then_some(neighbor as usize)
        })
    }

    fn upper_neighbors(&self, index: usize, level: u8) -> impl Iterator<Item = usize> + '_ {
        let node_level = self.level_at(index);
        let empty = 0..0;
        let range = if level == 0 || level > node_level {
            empty
        } else {
            let start = self.upper_offset_at(index) as usize + (level as usize - 1) * HNSW_UPPER_M;
            start..start + HNSW_UPPER_M
        };

        range.filter_map(move |position| {
            let neighbor = read_u32_at(
                &self.mmap,
                self.upper_neighbors_offset + position * size_of::<u32>(),
            );
            (neighbor != EMPTY_NEIGHBOR).then_some(neighbor as usize)
        })
    }

    fn level_at(&self, index: usize) -> u8 {
        self.mmap[self.levels_offset + index]
    }

    fn upper_offset_at(&self, index: usize) -> u64 {
        read_u64_at(
            &self.mmap,
            self.upper_offsets_offset + index * size_of::<u64>(),
        )
    }

    fn level0_neighbor_offset(&self, index: usize, slot: usize) -> usize {
        self.level0_offset + (index * HNSW_M + slot) * size_of::<u32>()
    }

    fn reference_vector_offset(&self, index: usize) -> usize {
        debug_assert!(index < self.reference_count);
        self.vectors_offset + index * VECTOR_LEN
    }

    fn reference_vector_ptr(&self, index: usize) -> *const i16 {
        unsafe {
            self.mmap
                .as_ptr()
                .add(self.reference_vector_offset(index))
                .cast::<i16>()
        }
    }
}

#[derive(Debug)]
pub struct HnswLayout {
    pub reference_count: usize,
    pub entrypoint: usize,
    pub max_level: u8,
    pub upper_edge_count: usize,
    pub vectors_offset: usize,
    pub fraud_offset: usize,
    pub levels_offset: usize,
    pub level0_offset: usize,
    pub upper_offsets_offset: usize,
    pub upper_neighbors_offset: usize,
}

impl HnswLayout {
    pub fn read(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HNSW_HEADER_LEN {
            bail!("invalid HNSW binary: file is smaller than header");
        }

        if &bytes[..HNSW_MAGIC.len()] != HNSW_MAGIC {
            bail!("invalid HNSW binary: bad magic");
        }

        let reference_count = read_u64_at(bytes, HNSW_MAGIC.len()) as usize;
        let entrypoint_offset = HNSW_MAGIC.len() + size_of::<u64>();
        let entrypoint = read_u32_at(bytes, entrypoint_offset) as usize;
        let max_level = bytes[entrypoint_offset + size_of::<u32>()];
        let upper_edge_count_offset = entrypoint_offset + size_of::<u32>() + size_of::<u8>() + 3;
        let upper_edge_count = read_u64_at(bytes, upper_edge_count_offset) as usize;

        if reference_count == 0 {
            bail!("invalid HNSW binary: no references");
        }

        if entrypoint >= reference_count {
            bail!("invalid HNSW binary: entrypoint out of bounds");
        }

        let vectors_offset = HNSW_HEADER_LEN;
        let fraud_offset = vectors_offset + reference_count * VECTOR_LEN;
        let levels_offset = fraud_offset + reference_count.div_ceil(8);
        let level0_offset = levels_offset + reference_count;
        let upper_offsets_offset = level0_offset + reference_count * HNSW_M * size_of::<u32>();
        let upper_neighbors_offset =
            upper_offsets_offset + (reference_count + 1) * size_of::<u64>();
        let expected_len = upper_neighbors_offset + upper_edge_count * size_of::<u32>();

        if bytes.len() != expected_len {
            bail!(
                "invalid HNSW binary: expected {} bytes, got {} bytes",
                expected_len,
                bytes.len()
            );
        }

        let last_upper_offset = read_u64_at(
            bytes,
            upper_offsets_offset + reference_count * size_of::<u64>(),
        ) as usize;
        if last_upper_offset != upper_edge_count {
            bail!(
                "invalid HNSW binary: last upper offset {} does not match upper edge count {}",
                last_upper_offset,
                upper_edge_count
            );
        }

        Ok(Self {
            reference_count,
            entrypoint,
            max_level,
            upper_edge_count,
            vectors_offset,
            fraud_offset,
            levels_offset,
            level0_offset,
            upper_offsets_offset,
            upper_neighbors_offset,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HeapCandidate {
    distance: u64,
    index: usize,
}

impl Ord for HeapCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for HeapCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct TopNearest {
    candidates: Vec<NearestCandidate>,
    capacity: usize,
}

impl TopNearest {
    fn new(capacity: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn is_full(&self) -> bool {
        self.candidates.len() >= self.capacity
    }

    fn current_worst_distance(&self) -> u64 {
        if !self.is_full() {
            return u64::MAX;
        }

        self.candidates
            .iter()
            .map(|candidate| candidate.distance)
            .max()
            .unwrap_or(u64::MAX)
    }

    fn add(&mut self, candidate: NearestCandidate) {
        if self.capacity == 0 || self.contains(candidate.index) {
            return;
        }

        if self.candidates.len() < self.capacity {
            self.candidates.push(candidate);
            return;
        }

        let Some((slot, worst)) = self
            .candidates
            .iter()
            .enumerate()
            .max_by_key(|(_, candidate)| candidate.distance)
        else {
            return;
        };

        if candidate.distance < worst.distance {
            self.candidates[slot] = candidate;
        }
    }

    fn contains(&self, index: usize) -> bool {
        self.candidates
            .iter()
            .any(|candidate| candidate.index == index)
    }

    fn into_sorted(mut self) -> Vec<NearestCandidate> {
        self.candidates
            .sort_unstable_by_key(|candidate| (candidate.distance, candidate.index));
        self.candidates
    }
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

pub fn distance2_limited(a: &ReferenceVector, b: &ReferenceVector, limit: u64) -> u64 {
    let mut distance = distance2_first8_vector(a, b);

    if distance >= limit {
        return distance;
    }

    distance += distance2_tail6_vector(a, b);
    distance
}

fn distance2_ptr_limited(reference: *const i16, vector: &ReferenceVector, limit: u64) -> u64 {
    let mut distance = distance2_first8_ptr(reference, vector);

    if distance >= limit {
        return distance;
    }

    distance += distance2_tail6_ptr(reference, vector);
    distance
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
fn distance2_first8_ptr(reference: *const i16, vector: &ReferenceVector) -> u64 {
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

fn distance2_tail6_vector(a: &ReferenceVector, b: &ReferenceVector) -> u64 {
    #[cfg(test)]
    TAIL_EVALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    a[8..]
        .iter()
        .zip(&b[8..])
        .map(|(left, right)| {
            let delta = i64::from(*left) - i64::from(*right);
            (delta * delta) as u64
        })
        .sum()
}

fn distance2_tail6_ptr(reference: *const i16, vector: &ReferenceVector) -> u64 {
    #[cfg(test)]
    TAIL_EVALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let mut distance = 0;

    for dimension in 8..VECTOR_DIMENSIONS {
        let left = i64::from(vector[dimension]);
        let right = i64::from(unsafe { *reference.add(dimension) });
        let delta = left - right;
        distance += (delta * delta) as u64;
    }

    distance
}

#[cfg(test)]
static TAIL_EVALS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::Ordering;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn append_vector(bytes: &mut Vec<u8>, vector: &ReferenceVector) {
        for value in vector {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn sample_hnsw_bytes(
        vectors: &[ReferenceVector],
        fraud_bits: &[u8],
        levels: &[u8],
        level0: &[[u32; HNSW_M]],
        upper: &[Vec<u32>],
        entrypoint: u32,
        max_level: u8,
    ) -> Vec<u8> {
        let upper_edge_count = upper.iter().map(Vec::len).sum::<usize>();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(HNSW_MAGIC);
        bytes.extend_from_slice(&(vectors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&entrypoint.to_le_bytes());
        bytes.push(max_level);
        bytes.extend_from_slice(&[0; 3]);
        bytes.extend_from_slice(&(upper_edge_count as u64).to_le_bytes());

        for vector in vectors {
            append_vector(&mut bytes, vector);
        }

        bytes.extend_from_slice(fraud_bits);
        bytes.extend_from_slice(levels);

        for neighbors in level0 {
            for neighbor in neighbors {
                bytes.extend_from_slice(&neighbor.to_le_bytes());
            }
        }

        let mut offset = 0_u64;
        bytes.extend_from_slice(&offset.to_le_bytes());
        for neighbors in upper {
            offset += neighbors.len() as u64;
            bytes.extend_from_slice(&offset.to_le_bytes());
        }

        for neighbors in upper {
            for neighbor in neighbors {
                bytes.extend_from_slice(&neighbor.to_le_bytes());
            }
        }

        bytes
    }

    fn mmap_bytes(bytes: &[u8]) -> Mmap {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "rinha-hnsw-test-{}-{nanos}.bin",
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

    fn sample_index(bytes: Vec<u8>) -> HnswIndex {
        let mmap = mmap_bytes(&bytes);
        let layout = HnswLayout::read(&mmap).unwrap();
        HnswIndex {
            mmap,
            reference_count: layout.reference_count,
            entrypoint: layout.entrypoint,
            max_level: layout.max_level,
            vectors_offset: layout.vectors_offset,
            fraud_offset: layout.fraud_offset,
            levels_offset: layout.levels_offset,
            level0_offset: layout.level0_offset,
            upper_offsets_offset: layout.upper_offsets_offset,
            upper_neighbors_offset: layout.upper_neighbors_offset,
        }
    }

    #[test]
    fn parser_accepts_hnsw_magic() {
        let vectors = [[0_i16; VECTOR_DIMENSIONS]];
        let levels = [0_u8];
        let level0 = [[EMPTY_NEIGHBOR; HNSW_M]];
        let upper = [Vec::new()];
        let bytes = sample_hnsw_bytes(&vectors, &[0], &levels, &level0, &upper, 0, 0);

        let layout = HnswLayout::read(&bytes).unwrap();

        assert_eq!(layout.reference_count, 1);
        assert_eq!(layout.entrypoint, 0);
    }

    #[test]
    fn parser_rejects_old_magic() {
        let vectors = [[0_i16; VECTOR_DIMENSIONS]];
        let levels = [0_u8];
        let level0 = [[EMPTY_NEIGHBOR; HNSW_M]];
        let upper = [Vec::new()];
        let mut bytes = sample_hnsw_bytes(&vectors, &[0], &levels, &level0, &upper, 0, 0);
        let old_magic = [b"R26".as_slice(), b"IFL04".as_slice()].concat();
        bytes[..HNSW_MAGIC.len()].copy_from_slice(&old_magic);

        let error = HnswLayout::read(&bytes).unwrap_err().to_string();

        assert!(error.contains("bad magic"));
    }

    #[test]
    fn distance_uses_first8_plus_tail6() {
        let mut a = [0_i16; VECTOR_DIMENSIONS];
        let mut b = [0_i16; VECTOR_DIMENSIONS];
        a[0] = 1_000;
        b[0] = 4_000;
        a[8] = 2_000;
        b[8] = 6_000;
        a[13] = -1_000;
        b[13] = 1_000;

        assert_eq!(distance2_limited(&a, &b, u64::MAX), 29_000_000);
    }

    #[test]
    fn early_discard_skips_tail6() {
        let mut a = [0_i16; VECTOR_DIMENSIONS];
        let b = [0_i16; VECTOR_DIMENSIONS];
        a[0] = 10_000;

        TAIL_EVALS.store(0, Ordering::Relaxed);
        let distance = distance2_limited(&a, &b, 1);

        assert_eq!(distance, 100_000_000);
        assert_eq!(TAIL_EVALS.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn search_hnsw_returns_expected_top_on_manual_graph() {
        let mut vectors = [[0_i16; VECTOR_DIMENSIONS]; 6];
        for (index, vector) in vectors.iter_mut().enumerate() {
            vector[0] = (index as i16) * 100;
        }
        let levels = [0_u8; 6];
        let mut level0 = [[EMPTY_NEIGHBOR; HNSW_M]; 6];
        level0[0][..2].copy_from_slice(&[1, 2]);
        level0[1][..3].copy_from_slice(&[0, 2, 3]);
        level0[2][..4].copy_from_slice(&[0, 1, 3, 4]);
        level0[3][..4].copy_from_slice(&[1, 2, 4, 5]);
        level0[4][..3].copy_from_slice(&[2, 3, 5]);
        level0[5][..2].copy_from_slice(&[3, 4]);
        let upper = [(); 6].map(|_| Vec::new());
        let bytes = sample_hnsw_bytes(&vectors, &[0], &levels, &level0, &upper, 0, 0);
        let index = sample_index(bytes);
        let mut query = [0_i16; VECTOR_DIMENSIONS];
        query[0] = 250;

        let found = index.search_top(&query, 5);

        assert_eq!(
            found
                .iter()
                .map(|candidate| candidate.index)
                .collect::<Vec<_>>(),
            vec![2, 3, 1, 4, 0]
        );
    }

    #[test]
    fn empty_neighbor_slots_are_ignored() {
        let vectors = [[0_i16; VECTOR_DIMENSIONS], [100_i16; VECTOR_DIMENSIONS]];
        let levels = [0_u8; 2];
        let level0 = [[EMPTY_NEIGHBOR; HNSW_M], [EMPTY_NEIGHBOR; HNSW_M]];
        let upper = [Vec::new(), Vec::new()];
        let bytes = sample_hnsw_bytes(&vectors, &[0], &levels, &level0, &upper, 0, 0);
        let index = sample_index(bytes);

        let found = index.search_top(&[0; VECTOR_DIMENSIONS], 5);

        assert_eq!(found.len(), 1);
        assert_eq!(found[0].index, 0);
    }
}
