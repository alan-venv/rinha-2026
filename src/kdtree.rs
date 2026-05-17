use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use memmap2::{Advice, Mmap};

use crate::morton::MortonEntry;

#[cfg(not(target_arch = "x86_64"))]
compile_error!("kd-tree AVX2 path requires x86_64");

use std::arch::x86_64::{
    __m128i, __m256i, _mm_cvtsi128_si32, _mm_extract_epi32, _mm_loadu_si128, _mm_madd_epi16,
    _mm_sub_epi16, _mm256_add_epi16, _mm256_castsi256_si128, _mm256_extracti128_si256,
    _mm256_loadu_si256, _mm256_madd_epi16, _mm256_max_epi16, _mm256_setzero_si256,
    _mm256_sub_epi16,
};

const MAGIC: &[u8; 8] = b"RKDTREE1";
const VERSION: u32 = 4;
const HEADER_LEN: usize = 28;
const NODE_LEN: usize = 80;
const DIMENSIONS: usize = 16;
const VECTOR_LEN: usize = DIMENSIONS * 2;
const TOP_K: usize = 5;
const LEAF_SIZE: usize = 64;
const STACK_CAPACITY: usize = 128;
const NO_CHILD: u32 = u32::MAX;

pub struct KdTree {
    mmap: Mmap,
    nodes: usize,
    records: usize,
    vectors_offset: usize,
    labels_offset: usize,
}

#[derive(Clone, Copy)]
struct KdNode {
    min: [i16; DIMENSIONS],
    max: [i16; DIMENSIONS],
    left: u32,
    right: u32,
    start: u32,
    len: u32,
}

#[derive(Clone, Copy)]
struct Neighbor {
    distance: u64,
    index: u32,
    label: u8,
}

#[derive(Clone, Copy)]
struct KdRecord {
    vector: [i16; DIMENSIONS],
    label: u8,
}

pub struct KdTreeBuilder {
    nodes: Vec<KdNode>,
    indices: Vec<u32>,
    records: Vec<KdRecord>,
}

impl KdTree {
    pub fn load_default() -> io::Result<Self> {
        Self::load("resources/kdtree.bin")
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let input = File::open(path)?;
        let mmap = unsafe { Mmap::map(&input)? };

        if mmap.len() < HEADER_LEN || &mmap[..8] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid kd-tree index magic",
            ));
        }

        let version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported kd-tree index version",
            ));
        }

        let nodes = u64::from_le_bytes(mmap[12..20].try_into().unwrap()) as usize;
        let records = u64::from_le_bytes(mmap[20..28].try_into().unwrap()) as usize;
        let nodes_len = nodes.checked_mul(NODE_LEN).ok_or_else(invalid_length)?;
        let vectors_len = records.checked_mul(VECTOR_LEN).ok_or_else(invalid_length)?;
        let labels_len = records;
        let vectors_offset = HEADER_LEN
            .checked_add(nodes_len)
            .ok_or_else(invalid_length)?;
        let labels_offset = vectors_offset
            .checked_add(vectors_len)
            .ok_or_else(invalid_length)?;
        let expected_len = labels_offset
            .checked_add(labels_len)
            .ok_or_else(invalid_length)?;

        if mmap.len() != expected_len {
            return Err(invalid_length());
        }

        mmap.advise(Advice::WillNeed)?;
        warmup(&mmap);

        Ok(Self {
            mmap,
            nodes,
            records,
            vectors_offset,
            labels_offset,
        })
    }

    pub fn score(&self, vector: &[i16; DIMENSIONS]) -> f32 {
        let top = self.top5(vector);
        let frauds = top.iter().filter(|neighbor| neighbor.label == 1).count();
        frauds as f32 / TOP_K as f32
    }

    fn top5(&self, vector: &[i16; DIMENSIONS]) -> [Neighbor; TOP_K] {
        let mut top = [Neighbor {
            distance: u64::MAX,
            index: u32::MAX,
            label: 0,
        }; TOP_K];

        if self.nodes == 0 {
            return top;
        }

        let mut stack = [0_usize; STACK_CAPACITY];
        let mut stack_len = 1;

        while stack_len > 0 {
            stack_len -= 1;
            let node_index = stack[stack_len];
            let node = self.node_at(node_index);
            if lower_bound_l2(vector, &node) > top[TOP_K - 1].distance {
                continue;
            }

            if node.left == NO_CHILD {
                let start = node.start as usize;
                let end = start + node.len as usize;
                for record_index in start..end {
                    let candidate = self.vector_at(record_index);
                    let Some(distance) = (unsafe {
                        l2_distance_with_first_8_exit(vector, candidate, top[TOP_K - 1].distance)
                    }) else {
                        continue;
                    };

                    insert_neighbor(
                        &mut top,
                        Neighbor {
                            distance,
                            index: record_index as u32,
                            label: self.label_at(record_index),
                        },
                    );
                }
                continue;
            }

            let left = node.left as usize;
            let right = node.right as usize;
            let left_node = self.node_at(left);
            let right_node = self.node_at(right);
            let left_bound = lower_bound_l2(vector, &left_node);
            let right_bound = lower_bound_l2(vector, &right_node);
            let worst = top[TOP_K - 1].distance;

            if left_bound <= right_bound {
                if right_bound <= worst {
                    push_stack(&mut stack, &mut stack_len, right);
                }
                if left_bound <= worst {
                    push_stack(&mut stack, &mut stack_len, left);
                }
            } else {
                if left_bound <= worst {
                    push_stack(&mut stack, &mut stack_len, left);
                }
                if right_bound <= worst {
                    push_stack(&mut stack, &mut stack_len, right);
                }
            }
        }

        top
    }

    fn node_at(&self, index: usize) -> KdNode {
        let start = HEADER_LEN + (index * NODE_LEN);
        decode_node(&self.mmap[start..start + NODE_LEN])
    }

    fn vector_at(&self, index: usize) -> &[u8] {
        debug_assert!(index < self.records);
        let start = self.vectors_offset + (index * VECTOR_LEN);
        &self.mmap[start..start + VECTOR_LEN]
    }

    fn label_at(&self, index: usize) -> u8 {
        debug_assert!(index < self.records);
        self.mmap[self.labels_offset + index]
    }
}

impl KdTreeBuilder {
    pub fn build(entries: &[MortonEntry]) -> io::Result<Self> {
        if entries.len() > u32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "too many kd-tree entries",
            ));
        }

        let mut tree = Self {
            nodes: Vec::new(),
            indices: (0..entries.len() as u32).collect(),
            records: Vec::new(),
        };

        let len = tree.indices.len();
        if len > 0 {
            tree.build_node(entries, 0, len)?;
        }
        tree.records = tree
            .indices
            .iter()
            .map(|&entry_index| {
                let entry = entries[entry_index as usize];
                KdRecord {
                    vector: entry.vector,
                    label: entry.label,
                }
            })
            .collect();

        Ok(tree)
    }

    pub fn write(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut output = BufWriter::new(File::create(path)?);
        output.write_all(MAGIC)?;
        output.write_all(&VERSION.to_le_bytes())?;
        output.write_all(&(self.nodes.len() as u64).to_le_bytes())?;
        output.write_all(&(self.records.len() as u64).to_le_bytes())?;

        for node in &self.nodes {
            for value in node.min {
                output.write_all(&value.to_le_bytes())?;
            }
            for value in node.max {
                output.write_all(&value.to_le_bytes())?;
            }
            output.write_all(&node.left.to_le_bytes())?;
            output.write_all(&node.right.to_le_bytes())?;
            output.write_all(&node.start.to_le_bytes())?;
            output.write_all(&node.len.to_le_bytes())?;
        }

        for record in &self.records {
            for value in record.vector {
                output.write_all(&value.to_le_bytes())?;
            }
        }

        for record in &self.records {
            output.write_all(&[record.label])?;
        }

        output.flush()
    }

    pub fn nodes_len(&self) -> usize {
        self.nodes.len()
    }

    pub fn records_len(&self) -> usize {
        self.records.len()
    }

    fn build_node(&mut self, entries: &[MortonEntry], start: usize, len: usize) -> io::Result<u32> {
        let node_index = self.nodes.len();
        if node_index > u32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "too many kd-tree nodes",
            ));
        }

        let (min, max) = bounds(entries, &self.indices[start..start + len]);
        self.nodes.push(KdNode {
            min,
            max,
            left: NO_CHILD,
            right: NO_CHILD,
            start: start as u32,
            len: len as u32,
        });

        if len <= LEAF_SIZE {
            return Ok(node_index as u32);
        }

        let split_dimension = widest_dimension(&min, &max);
        let middle = len / 2;
        self.indices[start..start + len].select_nth_unstable_by_key(middle, |entry_index| {
            entries[*entry_index as usize].vector[split_dimension]
        });

        let left = self.build_node(entries, start, middle)?;
        let right = self.build_node(entries, start + middle, len - middle)?;
        self.nodes[node_index].left = left;
        self.nodes[node_index].right = right;

        Ok(node_index as u32)
    }
}

fn bounds(entries: &[MortonEntry], indices: &[u32]) -> ([i16; DIMENSIONS], [i16; DIMENSIONS]) {
    let mut min = [i16::MAX; DIMENSIONS];
    let mut max = [i16::MIN; DIMENSIONS];

    for &entry_index in indices {
        let vector = &entries[entry_index as usize].vector;
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

fn decode_node(raw: &[u8]) -> KdNode {
    let mut min = [0_i16; DIMENSIONS];
    let mut max = [0_i16; DIMENSIONS];
    let mut offset = 0;

    for value in &mut min {
        *value = i16::from_le_bytes(raw[offset..offset + 2].try_into().unwrap());
        offset += 2;
    }

    for value in &mut max {
        *value = i16::from_le_bytes(raw[offset..offset + 2].try_into().unwrap());
        offset += 2;
    }

    let left = u32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let right = u32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let start = u32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let len = u32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap());

    KdNode {
        min,
        max,
        left,
        right,
        start,
        len,
    }
}

fn lower_bound_l2(vector: &[i16; DIMENSIONS], node: &KdNode) -> u64 {
    unsafe { lower_bound_l2_avx2(vector, node) }
}

fn push_stack(stack: &mut [usize; STACK_CAPACITY], len: &mut usize, value: usize) {
    debug_assert!(*len < STACK_CAPACITY);
    stack[*len] = value;
    *len += 1;
}

fn insert_neighbor(top: &mut [Neighbor; TOP_K], neighbor: Neighbor) {
    if !neighbor_better(neighbor, top[TOP_K - 1]) {
        return;
    }

    let mut output = TOP_K - 1;
    while output > 0 && neighbor_better(neighbor, top[output - 1]) {
        top[output] = top[output - 1];
        output -= 1;
    }
    top[output] = neighbor;
}

fn neighbor_better(left: Neighbor, right: Neighbor) -> bool {
    left.distance < right.distance || (left.distance == right.distance && left.index < right.index)
}

#[inline(always)]
unsafe fn l2_distance_with_first_8_exit(
    left: &[i16; DIMENSIONS],
    right: &[u8],
    worst: u64,
) -> Option<u64> {
    debug_assert_eq!(right.len(), VECTOR_LEN);

    let first = unsafe { l2_distance_8_i16(left.as_ptr(), right.as_ptr()) };
    if first > worst {
        return None;
    }

    let last = unsafe { l2_distance_8_i16(left.as_ptr().add(8), right.as_ptr().add(16)) };
    Some(first + last)
}

#[inline(always)]
unsafe fn l2_distance_8_i16(left: *const i16, right: *const u8) -> u64 {
    let left = unsafe { _mm_loadu_si128(left as *const __m128i) };
    let right = unsafe { _mm_loadu_si128(right as *const __m128i) };
    let diff = unsafe { _mm_sub_epi16(left, right) };
    let squares = unsafe { _mm_madd_epi16(diff, diff) };
    unsafe { horizontal_sum_i32x4(squares) }
}

#[inline(always)]
unsafe fn lower_bound_l2_avx2(vector: &[i16; DIMENSIONS], node: &KdNode) -> u64 {
    let value = unsafe { _mm256_loadu_si256(vector.as_ptr() as *const __m256i) };
    let min = unsafe { _mm256_loadu_si256(node.min.as_ptr() as *const __m256i) };
    let max = unsafe { _mm256_loadu_si256(node.max.as_ptr() as *const __m256i) };
    let zero = unsafe { _mm256_setzero_si256() };

    let below = unsafe { _mm256_max_epi16(_mm256_sub_epi16(min, value), zero) };
    let above = unsafe { _mm256_max_epi16(_mm256_sub_epi16(value, max), zero) };
    let diff = unsafe { _mm256_add_epi16(below, above) };
    let squares = unsafe { _mm256_madd_epi16(diff, diff) };

    unsafe { horizontal_sum_i32x8(squares) }
}

#[inline(always)]
unsafe fn horizontal_sum_i32x8(values: __m256i) -> u64 {
    let low = unsafe { _mm256_castsi256_si128(values) };
    let high = unsafe { _mm256_extracti128_si256::<1>(values) };
    unsafe { horizontal_sum_i32x4(low) + horizontal_sum_i32x4(high) }
}

#[inline(always)]
unsafe fn horizontal_sum_i32x4(values: __m128i) -> u64 {
    let lane0 = unsafe { _mm_cvtsi128_si32(values) } as u64;
    let lane1 = unsafe { _mm_extract_epi32::<1>(values) } as u64;
    let lane2 = unsafe { _mm_extract_epi32::<2>(values) } as u64;
    let lane3 = unsafe { _mm_extract_epi32::<3>(values) } as u64;
    lane0 + lane1 + lane2 + lane3
}

fn invalid_length() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid kd-tree index length")
}

fn warmup(mmap: &Mmap) {
    let mut checksum = 0_u8;
    let mut offset = 0;

    while offset < mmap.len() {
        checksum ^= mmap[offset];
        offset += 4096;
    }

    if let Some(last) = mmap.last() {
        checksum ^= *last;
    }

    std::hint::black_box(checksum);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_bound_l2_matches_scalar_formula() {
        let vector = [
            -10_000, -8_000, -2_000, -1, 0, 1, 999, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000,
            8_000, 9_000, 10_000,
        ];
        let node = KdNode {
            min: [
                -9_000, -7_000, -3_000, -2, -1, 0, 1_000, 2_000, 2_500, 4_100, 5_000, 6_001, 7_100,
                7_000, 9_000, 9_500,
            ],
            max: [
                -8_500, -6_500, -1_000, 0, 1, 2, 2_000, 3_000, 3_500, 4_200, 5_100, 6_100, 7_200,
                7_500, 9_100, 9_700,
            ],
            left: NO_CHILD,
            right: NO_CHILD,
            start: 0,
            len: 0,
        };

        assert_eq!(
            lower_bound_l2(&vector, &node),
            lower_bound_l2_scalar(&vector, &node)
        );
    }

    fn lower_bound_l2_scalar(vector: &[i16; DIMENSIONS], node: &KdNode) -> u64 {
        let mut distance = 0_u64;

        for (dimension, value) in vector.iter().enumerate() {
            let difference = if *value < node.min[dimension] {
                node.min[dimension] as i32 - *value as i32
            } else if *value > node.max[dimension] {
                *value as i32 - node.max[dimension] as i32
            } else {
                0
            } as u64;

            distance += difference * difference;
        }

        distance
    }
}
