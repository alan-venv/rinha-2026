use std::env;
use std::fs::File;
use std::path::Path;

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

    fn for_each_primary_candidates<C, V>(
        &self,
        vector: &ReferenceVector,
        current_worst_top_distance: &mut C,
        visit: &mut V,
    ) where
        C: FnMut() -> u64,
        V: FnMut(usize, u64);
}

pub struct IndexedReferences {
    ivfs: IvfIndexes,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchCost {
    pub primary_list_candidates: usize,
    pub centroid_distance_ops: usize,
    pub flat_candidates: usize,
    pub flat_early_discards: usize,
    pub flat_full_distance_candidates: usize,
    pub flat_vector_dimensions_evaluated: usize,
}

impl SearchCost {
    #[allow(dead_code)]
    pub fn total_units(&self) -> usize {
        self.centroid_distance_ops + self.flat_vector_dimensions_evaluated
    }
}

pub fn load_references() -> Result<IndexedReferences> {
    let ivfs = IvfIndexes::load()?;

    Ok(IndexedReferences { ivfs })
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

    fn for_each_primary_candidates<C, V>(
        &self,
        vector: &ReferenceVector,
        current_worst_top_distance: &mut C,
        visit: &mut V,
    ) where
        C: FnMut() -> u64,
        V: FnMut(usize, u64),
    {
        self.ivfs
            .for_each_primary_candidates(vector, current_worst_top_distance, visit);
    }
}

impl IndexedReferences {
    #[allow(dead_code)]
    pub fn search_cost(&self, vector: &ReferenceVector) -> SearchCost {
        self.ivfs.search_cost(vector)
    }
}

impl ReferenceSource for [ReferenceRecord] {
    fn is_fraud(&self, index: usize) -> bool {
        self[index].is_fraud
    }

    fn for_each_primary_candidates<C, V>(
        &self,
        vector: &ReferenceVector,
        current_worst_top_distance: &mut C,
        visit: &mut V,
    ) where
        C: FnMut() -> u64,
        V: FnMut(usize, u64),
    {
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

    fn for_each_primary_candidates<C, V>(
        &self,
        vector: &ReferenceVector,
        current_worst_top_distance: &mut C,
        visit: &mut V,
    ) where
        C: FnMut() -> u64,
        V: FnMut(usize, u64),
    {
        self.as_slice()
            .for_each_primary_candidates(vector, current_worst_top_distance, visit);
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

fn is_candidate_distance_useful(distance: u64, limit: u64) -> bool {
    limit == u64::MAX || distance < limit
}

#[derive(Clone, Copy)]
struct DistanceEval {
    distance: u64,
    dimensions: usize,
    early_discard: bool,
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

impl IvfIndexes {
    fn load() -> Result<Self> {
        Ok(Self {
            primary: IvfIndex::load("IVF_PATH", IVF_PATH, None)?,
        })
    }

    fn is_fraud(&self, index: usize) -> bool {
        self.primary.is_fraud(index)
    }

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

        for centroid in centroids[..centroid_count].iter().copied() {
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
        let (centroids, centroid_count) = self.nearest_centroid_indexes(vector);
        let primary_list_candidates = centroids[..centroid_count]
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
            centroid_distance_ops: self.centroid_count * VECTOR_DIMENSIONS,
            ..SearchCost::default()
        };

        for centroid in centroids[..centroid_count].iter().copied() {
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
            let distance = self.centroid_distance2_at_limited(centroid, vector, limit);

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

    fn centroid_distance2_at_limited(
        &self,
        centroid: usize,
        vector: &ReferenceVector,
        limit: u64,
    ) -> u64 {
        distance2_mmap_at(
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

fn distance2_mmap_at(mmap: &Mmap, offset: usize, vector: &ReferenceVector, limit: u64) -> u64 {
    distance2_mmap_at_limited(mmap, offset, vector, limit).distance
}

fn distance2_mmap_at_limited(
    mmap: &Mmap,
    offset: usize,
    vector: &ReferenceVector,
    limit: u64,
) -> DistanceEval {
    let reference = unsafe { mmap.as_ptr().add(offset).cast::<i16>() };
    let mut distance = 0;
    let early_dimensions = FLAT_EARLY_DIMENSIONS.min(VECTOR_DIMENSIONS);
    let mut dimensions = 0;

    for (dimension, query_value) in vector.iter().take(early_dimensions).enumerate() {
        let reference_value = i16::from_le(unsafe { *reference.add(dimension) });
        let delta = i64::from(*query_value) - i64::from(reference_value);
        distance += (delta * delta) as u64;
        dimensions += 1;
    }

    if distance >= limit {
        return DistanceEval {
            distance,
            dimensions,
            early_discard: dimensions < VECTOR_DIMENSIONS,
        };
    }

    for (dimension, query_value) in vector.iter().enumerate().skip(early_dimensions) {
        let reference_value = i16::from_le(unsafe { *reference.add(dimension) });
        let delta = i64::from(*query_value) - i64::from(reference_value);
        distance += (delta * delta) as u64;
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
