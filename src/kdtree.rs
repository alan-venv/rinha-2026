use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use memmap2::{Advice, Mmap};

use crate::morton::{MortonEntry, MortonIndex};

const MAGIC: &[u8; 8] = b"RKDTREE1";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 28;
const NODE_LEN: usize = 72;
const DIMENSIONS: usize = 14;
const TOP_K: usize = 5;
const LEAF_SIZE: usize = 64;
const NO_CHILD: u32 = u32::MAX;

pub struct KdTree {
    mmap: Mmap,
    nodes: usize,
    indices: usize,
    indices_offset: usize,
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

pub struct KdTreeBuilder {
    nodes: Vec<KdNode>,
    indices: Vec<u32>,
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
        let indices = u64::from_le_bytes(mmap[20..28].try_into().unwrap()) as usize;
        let nodes_len = nodes.checked_mul(NODE_LEN).ok_or_else(invalid_length)?;
        let indices_len = indices.checked_mul(4).ok_or_else(invalid_length)?;
        let indices_offset = HEADER_LEN
            .checked_add(nodes_len)
            .ok_or_else(invalid_length)?;
        let expected_len = indices_offset
            .checked_add(indices_len)
            .ok_or_else(invalid_length)?;

        if mmap.len() != expected_len {
            return Err(invalid_length());
        }

        mmap.advise(Advice::WillNeed)?;
        warmup(&mmap);

        Ok(Self {
            mmap,
            nodes,
            indices,
            indices_offset,
        })
    }

    pub fn score(&self, index: &MortonIndex, vector: &[i16; DIMENSIONS]) -> f32 {
        let top = self.top5(index, vector);
        let frauds = top.iter().filter(|neighbor| neighbor.label == 1).count();
        frauds as f32 / TOP_K as f32
    }

    fn top5(&self, index: &MortonIndex, vector: &[i16; DIMENSIONS]) -> [Neighbor; TOP_K] {
        let mut top = [Neighbor {
            distance: u64::MAX,
            index: u32::MAX,
            label: 0,
        }; TOP_K];

        if self.nodes == 0 {
            return top;
        }

        let mut stack = vec![0_usize];
        while let Some(node_index) = stack.pop() {
            let node = self.node_at(node_index);
            if lower_bound_l2(vector, &node) > top[TOP_K - 1].distance {
                continue;
            }

            if node.left == NO_CHILD {
                let start = node.start as usize;
                let end = start + node.len as usize;
                for index_offset in start..end {
                    let entry_index = self.index_at(index_offset);
                    let entry = index.entry_at(entry_index as usize);
                    insert_neighbor(
                        &mut top,
                        Neighbor {
                            distance: l2_distance(vector, &entry.vector),
                            index: entry_index,
                            label: entry.label,
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
                    stack.push(right);
                }
                if left_bound <= worst {
                    stack.push(left);
                }
            } else {
                if left_bound <= worst {
                    stack.push(left);
                }
                if right_bound <= worst {
                    stack.push(right);
                }
            }
        }

        top
    }

    fn node_at(&self, index: usize) -> KdNode {
        let start = HEADER_LEN + (index * NODE_LEN);
        decode_node(&self.mmap[start..start + NODE_LEN])
    }

    fn index_at(&self, index: usize) -> u32 {
        debug_assert!(index < self.indices);
        let start = self.indices_offset + (index * 4);
        u32::from_le_bytes(self.mmap[start..start + 4].try_into().unwrap())
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
        };

        let len = tree.indices.len();
        if len > 0 {
            tree.build_node(entries, 0, len)?;
        }

        Ok(tree)
    }

    pub fn write(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut output = BufWriter::new(File::create(path)?);
        output.write_all(MAGIC)?;
        output.write_all(&VERSION.to_le_bytes())?;
        output.write_all(&(self.nodes.len() as u64).to_le_bytes())?;
        output.write_all(&(self.indices.len() as u64).to_le_bytes())?;

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

        for index in &self.indices {
            output.write_all(&index.to_le_bytes())?;
        }

        output.flush()
    }

    pub fn nodes_len(&self) -> usize {
        self.nodes.len()
    }

    pub fn indices_len(&self) -> usize {
        self.indices.len()
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

fn l2_distance(left: &[i16; DIMENSIONS], right: &[i16; DIMENSIONS]) -> u64 {
    let mut distance = 0_u64;

    for (left, right) in left.iter().zip(right) {
        let difference = (*left as i32 - *right as i32).unsigned_abs() as u64;
        distance += difference * difference;
    }

    distance
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
