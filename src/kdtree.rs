use crate::morton::MortonIndex;

const DIMENSIONS: usize = 14;
const TOP_K: usize = 5;
const LEAF_SIZE: usize = 64;
const NO_CHILD: usize = usize::MAX;

pub struct KdTree {
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

impl KdTree {
    pub fn build(index: &MortonIndex) -> Self {
        let mut tree = Self {
            nodes: Vec::new(),
            indices: (0..index.len() as u32).collect(),
        };

        let len = tree.indices.len();
        if len > 0 {
            tree.build_node(index, 0, len);
        }

        tree
    }

    pub fn score(&self, index: &MortonIndex, vector: &[i16; DIMENSIONS]) -> f32 {
        let top = self.top5(index, vector);
        let frauds = top.iter().filter(|neighbor| neighbor.label == 1).count();
        frauds as f32 / TOP_K as f32
    }

    fn build_node(&mut self, index: &MortonIndex, start: usize, len: usize) -> usize {
        let (min, max) = bounds(index, &self.indices[start..start + len]);
        let node_index = self.nodes.len();
        self.nodes.push(KdNode {
            min,
            max,
            left: NO_CHILD,
            right: NO_CHILD,
            start,
            len,
        });

        if len <= LEAF_SIZE {
            return node_index;
        }

        let split_dimension = widest_dimension(&min, &max);
        let middle = len / 2;
        self.indices[start..start + len].select_nth_unstable_by_key(middle, |entry_index| {
            index.entry_at(*entry_index as usize).vector[split_dimension]
        });

        let left = self.build_node(index, start, middle);
        let right = self.build_node(index, start + middle, len - middle);
        self.nodes[node_index].left = left;
        self.nodes[node_index].right = right;

        node_index
    }

    fn top5(&self, index: &MortonIndex, vector: &[i16; DIMENSIONS]) -> [Neighbor; TOP_K] {
        let mut top = [Neighbor {
            distance: u64::MAX,
            index: u32::MAX,
            label: 0,
        }; TOP_K];

        if self.nodes.is_empty() {
            return top;
        }

        let mut stack = vec![0_usize];
        while let Some(node_index) = stack.pop() {
            let node = &self.nodes[node_index];
            if lower_bound_l2(vector, node) > top[TOP_K - 1].distance {
                continue;
            }

            if node.left == NO_CHILD {
                for &entry_index in &self.indices[node.start..node.start + node.len] {
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

            let left_bound = lower_bound_l2(vector, &self.nodes[node.left]);
            let right_bound = lower_bound_l2(vector, &self.nodes[node.right]);
            let worst = top[TOP_K - 1].distance;

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

        top
    }
}

fn bounds(index: &MortonIndex, indices: &[u32]) -> ([i16; DIMENSIONS], [i16; DIMENSIONS]) {
    let mut min = [i16::MAX; DIMENSIONS];
    let mut max = [i16::MIN; DIMENSIONS];

    for &entry_index in indices {
        let entry = index.entry_at(entry_index as usize);
        for dimension in 0..DIMENSIONS {
            min[dimension] = min[dimension].min(entry.vector[dimension]);
            max[dimension] = max[dimension].max(entry.vector[dimension]);
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
