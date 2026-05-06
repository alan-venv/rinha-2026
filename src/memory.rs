use std::env;
use std::fs::File;
use std::path::Path;

use rinha::*;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;

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
    pub pq_table_ops: usize,
    pub candidate_pq_code_reads: usize,
}

impl SearchCost {
    #[allow(dead_code)]
    pub fn total_units(&self) -> usize {
        self.centroid_distance_ops + self.pq_table_ops + self.candidate_pq_code_reads
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

fn read_i16_at(bytes: &[u8], offset: usize) -> i16 {
    debug_assert!(offset + size_of::<i16>() <= bytes.len());
    unsafe { i16::from_le(std::ptr::read_unaligned(bytes.as_ptr().add(offset).cast())) }
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
    reference_count: usize,
    centroid_count: usize,
    centroids_offset: usize,
    codebooks_offset: usize,
    offsets_offset: usize,
    indices_offset: usize,
    codes_offset: usize,
    fraud_offset: Option<usize>,
}

struct IvfIndexes {
    primary: IvfIndex,
}

impl IvfIndexes {
    fn load() -> Result<Self> {
        Ok(Self {
            primary: IvfIndex::load("IVF_PATH", IVF_PATH, None, true)?,
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
    fn load(
        env_key: &str,
        default_path: &str,
        reference_count: Option<usize>,
        require_fraud_labels: bool,
    ) -> Result<Self> {
        let path = env::var(env_key).unwrap_or_else(|_| default_path.to_string());
        let input_file = File::open(Path::new(&path))
            .with_context(|| format!("failed to open required IVF index at {path}"))?;
        let mmap = unsafe { Mmap::map(&input_file) }?;
        let layout = IvfLayout::read(&mmap, reference_count, require_fraud_labels)?;

        Ok(Self {
            mmap,
            reference_count: layout.reference_count,
            centroid_count: layout.centroid_count,
            centroids_offset: layout.centroids_offset,
            codebooks_offset: layout.codebooks_offset,
            offsets_offset: layout.offsets_offset,
            indices_offset: layout.indices_offset,
            codes_offset: layout.codes_offset,
            fraud_offset: layout.fraud_offset,
        })
    }

    fn is_fraud(&self, index: usize) -> bool {
        debug_assert!(index < self.reference_count);
        let fraud_offset = self
            .fraud_offset
            .expect("primary IVF index must have embedded fraud labels");
        let byte = self.mmap[fraud_offset + index / 8];
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
        let centroid = self.nearest_centroid_index(vector);
        self.for_each_candidate_in_centroid(centroid, vector, current_worst_top_distance, visit);
    }

    #[allow(dead_code)]
    fn search_cost(&self, vector: &ReferenceVector) -> SearchCost {
        let centroid = self.nearest_centroid_index(vector);
        let start = self.candidate_list_boundary_at(centroid) as usize;
        let end = self.candidate_list_boundary_at(centroid + 1) as usize;
        let primary_list_candidates = end - start;

        SearchCost {
            primary_list_candidates,
            centroid_distance_ops: self.centroid_count * VECTOR_DIMENSIONS,
            pq_table_ops: PQ_LAYOUT.0 * PQ_CODEWORDS * PQ_LAYOUT.1,
            candidate_pq_code_reads: primary_list_candidates * PQ_LAYOUT.0,
        }
    }

    fn for_each_candidate_in_centroid<C, V>(
        &self,
        centroid: usize,
        vector: &ReferenceVector,
        current_worst_top_distance: &mut C,
        visit: &mut V,
    ) where
        C: FnMut() -> u64,
        V: FnMut(usize, u64),
    {
        let start = self.candidate_list_boundary_at(centroid) as usize;
        let end = self.candidate_list_boundary_at(centroid + 1) as usize;
        let distance_table = self.pq_distance_table(centroid, vector);

        for position in start..end {
            let max_useful_distance = current_worst_top_distance();
            let distance = self.candidate_pq_distance(position, &distance_table);

            if is_candidate_distance_useful(distance, max_useful_distance) {
                let index = self.candidate_index_at(position) as usize;
                visit(index, distance);
            }
        }
    }

    fn nearest_centroid_index(&self, vector: &ReferenceVector) -> usize {
        let mut nearest_index = 0;
        let mut nearest_distance = u64::MAX;

        for centroid in 0..self.centroid_count {
            let distance = self.centroid_distance2_at(centroid, vector);

            if distance < nearest_distance {
                nearest_index = centroid;
                nearest_distance = distance;
            }
        }

        nearest_index
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

    fn pq_distance_table(
        &self,
        centroid: usize,
        vector: &ReferenceVector,
    ) -> [[u64; PQ_CODEWORDS]; PQ_LAYOUT.0] {
        let mut table = [[0_u64; PQ_CODEWORDS]; PQ_LAYOUT.0];

        for subquantizer in 0..PQ_LAYOUT.0 {
            for codeword in 0..PQ_CODEWORDS {
                table[subquantizer][codeword] = self.query_residual_codeword_distance2(
                    centroid,
                    vector,
                    subquantizer,
                    codeword,
                );
            }
        }

        table
    }

    fn query_residual_codeword_distance2(
        &self,
        centroid: usize,
        vector: &ReferenceVector,
        subquantizer: usize,
        codeword: usize,
    ) -> u64 {
        let mut distance = 0;
        let dimension_start = subquantizer * PQ_LAYOUT.1;
        let centroid_offset = self.centroid_vector_offset(centroid);
        let codeword_offset = self.codeword_offset(subquantizer, codeword);

        for dimension in 0..PQ_LAYOUT.1 {
            let query = vector[dimension_start + dimension];
            let centroid_value = read_i16_at(
                &self.mmap,
                centroid_offset + (dimension_start + dimension) * size_of::<i16>(),
            );
            let codeword_value =
                read_i16_at(&self.mmap, codeword_offset + dimension * size_of::<i16>());
            let residual = i64::from(query) - i64::from(centroid_value);
            let delta = residual - i64::from(codeword_value);
            distance += (delta * delta) as u64;
        }

        distance
    }

    fn candidate_pq_distance(
        &self,
        position: usize,
        distance_table: &[[u64; PQ_CODEWORDS]; PQ_LAYOUT.0],
    ) -> u64 {
        let mut distance = 0;
        let code_offset = self.candidate_code_offset(position);

        for subquantizer in 0..PQ_LAYOUT.0 {
            let codeword = self.mmap[code_offset + subquantizer] as usize;
            distance += distance_table[subquantizer][codeword];
        }

        distance
    }

    fn centroid_vector_offset(&self, centroid: usize) -> usize {
        self.centroids_offset + centroid * VECTOR_LEN
    }

    fn codeword_offset(&self, subquantizer: usize, codeword: usize) -> usize {
        self.codebooks_offset
            + (subquantizer * PQ_CODEWORDS + codeword) * PQ_LAYOUT.1 * size_of::<i16>()
    }

    fn candidate_list_boundary_offset(&self, centroid: usize) -> usize {
        self.offsets_offset + centroid * size_of::<u32>()
    }

    fn candidate_index_offset(&self, position: usize) -> usize {
        self.indices_offset + position * size_of::<u32>()
    }

    fn candidate_code_offset(&self, position: usize) -> usize {
        self.codes_offset + position * PQ_LAYOUT.0
    }
}

#[derive(Debug)]
struct IvfLayout {
    reference_count: usize,
    centroid_count: usize,
    centroids_offset: usize,
    codebooks_offset: usize,
    offsets_offset: usize,
    indices_offset: usize,
    codes_offset: usize,
    fraud_offset: Option<usize>,
}

impl IvfLayout {
    fn read(
        bytes: &[u8],
        expected_reference_count: Option<usize>,
        require_fraud_labels: bool,
    ) -> Result<Self> {
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
        let pq_subquantizers_offset = index_count_offset + size_of::<u64>();
        let pq_subquantizers = read_u32_at(bytes, pq_subquantizers_offset) as usize;
        let pq_codewords_offset = pq_subquantizers_offset + size_of::<u32>();
        let pq_codewords = read_u32_at(bytes, pq_codewords_offset) as usize;
        let pq_dimensions_offset = pq_codewords_offset + size_of::<u32>();
        let pq_dimensions = read_u32_at(bytes, pq_dimensions_offset) as usize;

        if pq_subquantizers != PQ_LAYOUT.0
            || pq_codewords != PQ_CODEWORDS
            || pq_dimensions != PQ_LAYOUT.1
        {
            bail!(
                "invalid IVF binary: expected PQ {}x{}x{}, got {}x{}x{}",
                PQ_LAYOUT.0,
                PQ_CODEWORDS,
                PQ_LAYOUT.1,
                pq_subquantizers,
                pq_codewords,
                pq_dimensions
            );
        }

        let centroids_offset = IVF_HEADER_LEN;
        let codebooks_offset = centroids_offset + centroid_count * VECTOR_LEN;
        let offsets_offset =
            codebooks_offset + PQ_LAYOUT.0 * PQ_CODEWORDS * PQ_LAYOUT.1 * size_of::<i16>();
        let indices_offset = offsets_offset + (centroid_count + 1) * size_of::<u32>();
        let codes_offset = indices_offset + index_count * size_of::<u32>();
        let fraud_offset = codes_offset + index_count * PQ_LAYOUT.0;
        let expected_len = fraud_offset
            + if require_fraud_labels {
                count.div_ceil(8)
            } else {
                0
            };

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
            codebooks_offset,
            offsets_offset,
            indices_offset,
            codes_offset,
            fraud_offset: require_fraud_labels.then_some(fraud_offset),
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
        bytes.extend_from_slice(&(PQ_LAYOUT.0 as u32).to_le_bytes());
        bytes.extend_from_slice(&(PQ_CODEWORDS as u32).to_le_bytes());
        bytes.extend_from_slice(&(PQ_LAYOUT.1 as u32).to_le_bytes());

        for _ in 0..centroid_count * VECTOR_DIMENSIONS {
            bytes.extend_from_slice(&0_i16.to_le_bytes());
        }

        for _ in 0..PQ_LAYOUT.0 * PQ_CODEWORDS * PQ_LAYOUT.1 {
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

        bytes.resize(bytes.len() + entry_count * PQ_LAYOUT.0, 0);
        bytes
    }

    fn mmap_bytes(bytes: &[u8]) -> Mmap {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "rinha-ivfpq-test-{}-{nanos}.bin",
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
    fn reads_ivf_pq_header_and_offsets() {
        let bytes = sample_ivf_bytes(3, 1, 3);
        let layout = IvfLayout::read(&bytes, Some(3), false).unwrap();

        assert_eq!(layout.reference_count, 3);
        assert_eq!(layout.centroid_count, 1);
        assert_eq!(layout.centroids_offset, IVF_HEADER_LEN);
        assert_eq!(
            layout.indices_offset,
            layout.offsets_offset + 2 * size_of::<u32>()
        );
        assert_eq!(
            layout.codes_offset,
            layout.indices_offset + 3 * size_of::<u32>()
        );
    }

    #[test]
    fn reads_embedded_fraud_labels() {
        let mut bytes = sample_ivf_bytes(3, 1, 3);
        bytes.push(0b0000_0101);
        let mmap = mmap_bytes(&bytes);
        let layout = IvfLayout::read(&mmap, Some(3), true).unwrap();
        let index = IvfIndex {
            mmap,
            reference_count: layout.reference_count,
            centroid_count: layout.centroid_count,
            centroids_offset: layout.centroids_offset,
            codebooks_offset: layout.codebooks_offset,
            offsets_offset: layout.offsets_offset,
            indices_offset: layout.indices_offset,
            codes_offset: layout.codes_offset,
            fraud_offset: layout.fraud_offset,
        };

        assert!(index.is_fraud(0));
        assert!(!index.is_fraud(1));
        assert!(index.is_fraud(2));
    }

    #[test]
    fn rejects_old_ivf_flat_magic() {
        let mut bytes = sample_ivf_bytes(3, 1, 3);
        bytes[..IVF_MAGIC.len()].copy_from_slice(b"R26IVF02");

        let error = IvfLayout::read(&bytes, Some(3), false)
            .unwrap_err()
            .to_string();

        assert!(error.contains("bad magic"));
    }

    #[test]
    fn rejects_invalid_last_offset() {
        let mut bytes = sample_ivf_bytes(3, 1, 3);
        let offsets_offset = IVF_HEADER_LEN
            + VECTOR_LEN
            + PQ_LAYOUT.0 * PQ_CODEWORDS * PQ_LAYOUT.1 * size_of::<i16>();
        bytes[offsets_offset + size_of::<u32>()..offsets_offset + 2 * size_of::<u32>()]
            .copy_from_slice(&2_u32.to_le_bytes());

        let error = IvfLayout::read(&bytes, Some(3), false)
            .unwrap_err()
            .to_string();

        assert!(error.contains("last offset"));
    }

    #[test]
    fn visits_candidates_with_pq_distances() {
        let bytes = sample_ivf_bytes(3, 1, 3);
        let mmap = mmap_bytes(&bytes);
        let layout = IvfLayout::read(&mmap, Some(3), false).unwrap();
        let index = IvfIndex {
            mmap,
            reference_count: layout.reference_count,
            centroid_count: layout.centroid_count,
            centroids_offset: layout.centroids_offset,
            codebooks_offset: layout.codebooks_offset,
            offsets_offset: layout.offsets_offset,
            indices_offset: layout.indices_offset,
            codes_offset: layout.codes_offset,
            fraud_offset: layout.fraud_offset,
        };
        let mut visited = Vec::new();

        index.for_each_candidates(
            &[0; VECTOR_DIMENSIONS],
            &mut || u64::MAX,
            &mut |index, distance| visited.push((index, distance)),
        );

        assert_eq!(visited, vec![(2, 0), (0, 0), (1, 0)]);
    }
}
