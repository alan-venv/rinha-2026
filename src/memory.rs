use std::env;
use std::fs::File;
use std::path::Path;

use rinha::*;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;

const REFERENCES_PATH: &str = "resources/references.bin";
const IVF_PATH: &str = "resources/ivf.bin";
const IVF_FRAUD_PATH: &str = "resources/ivf-fraud.bin";
const IVF_LEGIT_PATH: &str = "resources/ivf-legit.bin";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceRecord {
    pub vector: ReferenceVector,
    pub is_fraud: bool,
}

pub trait ReferenceSource {
    fn is_fraud(&self, index: usize) -> bool;

    fn for_each_primary_candidate_probes(
        &self,
        vector: &ReferenceVector,
        probes: usize,
        current_worst_top_distance: &mut dyn FnMut() -> u64,
        visit: &mut dyn FnMut(usize, u64),
    );

    fn for_each_auxiliary_candidate_probes(
        &self,
        _vector: &ReferenceVector,
        _probes: usize,
        _current_worst_top_distance: &mut dyn FnMut() -> u64,
        _visit: &mut dyn FnMut(usize, u64),
    ) {
    }
}

pub struct IndexedReferences {
    store: ReferenceStore,
    ivfs: IvfIndexes,
}

struct ReferenceStore {
    mmap: Mmap,
    count: usize,
    fraud_offset: usize,
}

pub fn load_references() -> Result<IndexedReferences> {
    let mmap = map_file(Path::new(REFERENCES_PATH))?;
    let layout = ReferenceLayout::read(&mmap)?;

    let ivfs = IvfIndexes::load(layout.count)?;

    Ok(IndexedReferences {
        store: ReferenceStore {
            mmap,
            count: layout.count,
            fraud_offset: layout.fraud_offset,
        },
        ivfs,
    })
}

struct ReferenceLayout {
    count: usize,
    fraud_offset: usize,
}

impl ReferenceLayout {
    fn read(mmap: &Mmap) -> Result<Self> {
        if mmap.len() < REFERENCE_HEADER_LEN {
            bail!("invalid references binary: file is smaller than header");
        }

        if &mmap[..REFERENCE_MAGIC.len()] != REFERENCE_MAGIC {
            bail!("invalid references binary: bad magic");
        }

        let count = read_u64_at(mmap, REFERENCE_MAGIC.len()) as usize;
        let fraud_offset = REFERENCE_HEADER_LEN + count * VECTOR_LEN;
        let expected_len = fraud_offset + count.div_ceil(8);

        if mmap.len() != expected_len {
            bail!(
                "invalid references binary: expected {} bytes, got {} bytes",
                expected_len,
                mmap.len()
            );
        }

        Ok(Self {
            count,
            fraud_offset,
        })
    }
}

fn map_file(path: &Path) -> Result<Mmap> {
    let input_file = File::open(path)?;
    Ok(unsafe { Mmap::map(&input_file) }?)
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
        self.store.is_fraud(index)
    }

    fn for_each_primary_candidate_probes(
        &self,
        vector: &ReferenceVector,
        probes: usize,
        current_worst_top_distance: &mut dyn FnMut() -> u64,
        visit: &mut dyn FnMut(usize, u64),
    ) {
        self.ivfs.for_each_primary_candidate_probes(
            &self.store,
            vector,
            probes,
            current_worst_top_distance,
            visit,
        );
    }

    fn for_each_auxiliary_candidate_probes(
        &self,
        vector: &ReferenceVector,
        probes: usize,
        current_worst_top_distance: &mut dyn FnMut() -> u64,
        visit: &mut dyn FnMut(usize, u64),
    ) {
        self.ivfs.for_each_auxiliary_candidate_probes(
            &self.store,
            vector,
            probes,
            current_worst_top_distance,
            visit,
        );
    }
}

impl ReferenceStore {
    fn distance2_at_limited(&self, index: usize, vector: &ReferenceVector, limit: u64) -> u64 {
        debug_assert!(index < self.count);
        distance2_mmap_at(&self.mmap, self.vector_offset(index), vector, limit)
    }

    fn is_fraud(&self, index: usize) -> bool {
        debug_assert!(index < self.count);
        let byte = self.mmap[self.fraud_offset + index / 8];
        byte & (1 << (index % 8)) != 0
    }

    fn vector_offset(&self, index: usize) -> usize {
        REFERENCE_HEADER_LEN + index * VECTOR_LEN
    }
}

impl ReferenceSource for [ReferenceRecord] {
    fn is_fraud(&self, index: usize) -> bool {
        self[index].is_fraud
    }

    fn for_each_primary_candidate_probes(
        &self,
        vector: &ReferenceVector,
        probes: usize,
        current_worst_top_distance: &mut dyn FnMut() -> u64,
        visit: &mut dyn FnMut(usize, u64),
    ) {
        let _ = probes;
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

    fn for_each_primary_candidate_probes(
        &self,
        vector: &ReferenceVector,
        probes: usize,
        current_worst_top_distance: &mut dyn FnMut() -> u64,
        visit: &mut dyn FnMut(usize, u64),
    ) {
        self.as_slice().for_each_primary_candidate_probes(
            vector,
            probes,
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

fn distance2_mmap_at(mmap: &Mmap, offset: usize, vector: &ReferenceVector, limit: u64) -> u64 {
    let reference = unsafe { mmap.as_ptr().add(offset).cast::<i16>() };
    let mut distance = 0;

    for (dimension, query_value) in vector.iter().enumerate() {
        let reference_value = i16::from_le(unsafe { *reference.add(dimension) });
        let delta = i64::from(*query_value) - i64::from(reference_value);
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

struct IvfIndex {
    mmap: Mmap,
    centroid_count: usize,
    centroids_offset: usize,
    offsets_offset: usize,
    indices_offset: usize,
}

struct IvfIndexes {
    primary: IvfIndex,
    fraud: IvfIndex,
    legit: IvfIndex,
}

impl IvfIndexes {
    fn load(reference_count: usize) -> Result<Self> {
        Ok(Self {
            primary: IvfIndex::load("IVF_PATH", IVF_PATH, reference_count)?,
            fraud: IvfIndex::load("IVF_FRAUD_PATH", IVF_FRAUD_PATH, reference_count)?,
            legit: IvfIndex::load("IVF_LEGIT_PATH", IVF_LEGIT_PATH, reference_count)?,
        })
    }

    fn for_each_primary_candidate_probes(
        &self,
        references: &ReferenceStore,
        vector: &ReferenceVector,
        probes: usize,
        current_worst_top_distance: &mut dyn FnMut() -> u64,
        visit: &mut dyn FnMut(usize, u64),
    ) {
        self.primary.for_each_candidate_probes(
            references,
            vector,
            probes,
            current_worst_top_distance,
            visit,
        );
    }

    fn for_each_auxiliary_candidate_probes(
        &self,
        references: &ReferenceStore,
        vector: &ReferenceVector,
        probes: usize,
        current_worst_top_distance: &mut dyn FnMut() -> u64,
        visit: &mut dyn FnMut(usize, u64),
    ) {
        self.fraud.for_each_candidate_probes(
            references,
            vector,
            probes,
            current_worst_top_distance,
            visit,
        );
        self.legit.for_each_candidate_probes(
            references,
            vector,
            probes,
            current_worst_top_distance,
            visit,
        );
    }
}

impl IvfIndex {
    fn load(env_key: &str, default_path: &str, reference_count: usize) -> Result<Self> {
        let path = env::var(env_key).unwrap_or_else(|_| default_path.to_string());
        let input_file = File::open(Path::new(&path))
            .with_context(|| format!("failed to open required IVF index at {path}"))?;
        let mmap = unsafe { Mmap::map(&input_file) }?;
        let layout = IvfLayout::read(&mmap, reference_count)?;

        Ok(Self {
            mmap,
            centroid_count: layout.centroid_count,
            centroids_offset: layout.centroids_offset,
            offsets_offset: layout.offsets_offset,
            indices_offset: layout.indices_offset,
        })
    }

    fn for_each_candidate_probes(
        &self,
        references: &ReferenceStore,
        vector: &ReferenceVector,
        probes: usize,
        current_worst_top_distance: &mut dyn FnMut() -> u64,
        visit: &mut dyn FnMut(usize, u64),
    ) {
        let nearest_centroids = self.nearest_centroids(vector, probes);

        for centroid in nearest_centroids.indexes() {
            let start = self.candidate_list_boundary_at(centroid) as usize;
            let end = self.candidate_list_boundary_at(centroid + 1) as usize;

            for position in start..end {
                let index = self.candidate_index_at(position) as usize;
                let max_useful_distance = current_worst_top_distance();
                let distance = references.distance2_at_limited(index, vector, max_useful_distance);

                if is_candidate_distance_useful(distance, max_useful_distance) {
                    visit(index, distance);
                }
            }
        }
    }

    fn nearest_centroids(&self, vector: &ReferenceVector, max_probes: usize) -> NearestCentroids {
        let capacity = max_probes.min(self.centroid_count).min(IVF_INITIAL_PROBES);
        let mut nearest = NearestCentroids::new(capacity);

        if nearest.is_empty() {
            return nearest;
        }

        for centroid in 0..self.centroid_count {
            let distance = self.centroid_distance2_at(centroid, vector);
            nearest.try_insert(centroid, distance);
        }

        nearest.sort();
        nearest
    }

    fn centroid_distance2_at(&self, centroid: usize, vector: &ReferenceVector) -> u64 {
        distance2_mmap_at(
            &self.mmap,
            self.centroid_vector_offset(centroid),
            vector,
            u64::MAX,
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
}

struct IvfLayout {
    centroid_count: usize,
    centroids_offset: usize,
    offsets_offset: usize,
    indices_offset: usize,
}

impl IvfLayout {
    fn read(mmap: &Mmap, reference_count: usize) -> Result<Self> {
        if mmap.len() < IVF_HEADER_LEN {
            bail!("invalid IVF binary: file is smaller than header");
        }

        if &mmap[..IVF_MAGIC.len()] != IVF_MAGIC {
            bail!("invalid IVF binary: bad magic");
        }

        let count = read_u64_at(mmap, IVF_MAGIC.len()) as usize;

        if count != reference_count {
            bail!(
                "invalid IVF binary: expected {} references, got {}",
                reference_count,
                count
            );
        }

        let centroid_count_offset = IVF_MAGIC.len() + size_of::<u64>();
        let centroid_count = read_u32_at(mmap, centroid_count_offset) as usize;

        if centroid_count == 0 {
            bail!("invalid IVF binary: no centroids");
        }

        let index_count_offset = centroid_count_offset + size_of::<u32>();
        let index_count = read_u64_at(mmap, index_count_offset) as usize;

        let centroids_offset = IVF_HEADER_LEN;
        let offsets_offset = centroids_offset + centroid_count * VECTOR_LEN;
        let indices_offset = offsets_offset + (centroid_count + 1) * size_of::<u32>();
        let expected_len = indices_offset + index_count * size_of::<u32>();

        if mmap.len() != expected_len {
            bail!(
                "invalid IVF binary: expected {} bytes, got {} bytes",
                expected_len,
                mmap.len()
            );
        }

        Ok(Self {
            centroid_count,
            centroids_offset,
            offsets_offset,
            indices_offset,
        })
    }
}

struct NearestCentroids {
    indexes: [usize; IVF_INITIAL_PROBES],
    distances: [u64; IVF_INITIAL_PROBES],
    capacity: usize,
    filled: usize,
    farthest_slot: usize,
}

impl NearestCentroids {
    fn new(capacity: usize) -> Self {
        Self {
            indexes: [0; IVF_INITIAL_PROBES],
            distances: [u64::MAX; IVF_INITIAL_PROBES],
            capacity,
            filled: 0,
            farthest_slot: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.capacity == 0
    }

    fn indexes(&self) -> impl Iterator<Item = usize> + '_ {
        self.indexes[..self.filled].iter().copied()
    }

    fn try_insert(&mut self, index: usize, distance: u64) {
        if self.filled < self.capacity {
            self.insert_at(self.filled, index, distance);

            if distance > self.distances[self.farthest_slot] {
                self.farthest_slot = self.filled;
            }

            self.filled += 1;
        } else if distance < self.distances[self.farthest_slot] {
            self.insert_at(self.farthest_slot, index, distance);
            self.farthest_slot = farthest_slot_in_slice(&self.distances[..self.capacity]);
        }
    }

    fn insert_at(&mut self, slot: usize, index: usize, distance: u64) {
        self.indexes[slot] = index;
        self.distances[slot] = distance;
    }

    fn sort(&mut self) {
        for slot in 1..self.filled {
            let index = self.indexes[slot];
            let distance = self.distances[slot];
            let mut cursor = slot;

            while cursor > 0 && distance < self.distances[cursor - 1] {
                self.indexes[cursor] = self.indexes[cursor - 1];
                self.distances[cursor] = self.distances[cursor - 1];
                cursor -= 1;
            }

            self.indexes[cursor] = index;
            self.distances[cursor] = distance;
        }
    }
}

fn farthest_slot_in_slice(distances: &[u64]) -> usize {
    let mut farthest_slot = 0;

    for slot in 1..distances.len() {
        if distances[slot] > distances[farthest_slot] {
            farthest_slot = slot;
        }
    }

    farthest_slot
}
