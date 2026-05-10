use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Result, bail};
use rinha::memory::{NearestCandidate, distance2_limited};
use rinha::*;

use crate::reference::ReferenceDataset;

pub struct IndexHnsw;

impl IndexHnsw {
    pub fn build(dataset: &ReferenceDataset) -> Result<()> {
        if dataset.len() == 0 {
            bail!("cannot build empty HNSW index");
        }

        println!(
            "building HNSW: references={}, m={}, upper_m={}, ef_construction={}",
            dataset.len(),
            HNSW_M,
            HNSW_UPPER_M,
            HNSW_EF_CONSTRUCTION,
        );

        let graph = HnswBuildGraph::build(dataset);
        Self::write_index(Path::new("resources/hnsw.bin"), dataset, &graph)?;

        println!("wrote resources/hnsw.bin");
        Ok(())
    }

    fn write_index(
        output: &Path,
        dataset: &ReferenceDataset,
        graph: &HnswBuildGraph,
    ) -> Result<()> {
        let output_file = File::create(output)?;
        let mut writer = BufWriter::new(output_file);
        let upper_edge_count = graph.upper.iter().map(Vec::len).sum::<usize>();

        writer.write_all(HNSW_MAGIC)?;
        writer.write_all(&(dataset.len() as u64).to_le_bytes())?;
        writer.write_all(&(graph.entrypoint as u32).to_le_bytes())?;
        writer.write_all(&[graph.max_level])?;
        writer.write_all(&[0; 3])?;
        writer.write_all(&(upper_edge_count as u64).to_le_bytes())?;

        for vector in &dataset.vectors {
            write_vector(&mut writer, vector)?;
        }

        writer.write_all(dataset.fraud_bits())?;
        writer.write_all(&graph.levels)?;

        for neighbors in &graph.level0 {
            for neighbor in neighbors {
                writer.write_all(&neighbor.to_le_bytes())?;
            }
        }

        let mut upper_offset = 0_u64;
        writer.write_all(&upper_offset.to_le_bytes())?;
        for neighbors in &graph.upper {
            upper_offset += neighbors.len() as u64;
            writer.write_all(&upper_offset.to_le_bytes())?;
        }

        for neighbors in &graph.upper {
            for neighbor in neighbors {
                writer.write_all(&neighbor.to_le_bytes())?;
            }
        }

        writer.flush()?;
        Ok(())
    }
}

struct HnswBuildGraph {
    levels: Vec<u8>,
    level0: Vec<[u32; HNSW_M]>,
    upper: Vec<Vec<u32>>,
    entrypoint: usize,
    max_level: u8,
    visited: Vec<u32>,
    visit_mark: u32,
}

impl HnswBuildGraph {
    fn build(dataset: &ReferenceDataset) -> Self {
        let mut graph = Self {
            levels: Vec::with_capacity(dataset.len()),
            level0: Vec::with_capacity(dataset.len()),
            upper: Vec::with_capacity(dataset.len()),
            entrypoint: 0,
            max_level: 0,
            visited: vec![0; dataset.len()],
            visit_mark: 0,
        };
        let mut rng = 0x9e37_79b9_7f4a_7c15_u64;

        for index in 0..dataset.len() {
            let level = random_level(&mut rng);
            graph.insert(dataset, index, level);

            if index > 0 && index % 100_000 == 0 {
                println!("inserted {index} references");
            }
        }

        graph
    }

    fn insert(&mut self, dataset: &ReferenceDataset, index: usize, level: u8) {
        self.levels.push(level);
        self.level0.push([EMPTY_NEIGHBOR; HNSW_M]);
        self.upper
            .push(vec![EMPTY_NEIGHBOR; level as usize * HNSW_UPPER_M]);

        if index == 0 {
            self.entrypoint = 0;
            self.max_level = level;
            return;
        }

        let vector = dataset.vector_at(index);
        let mut entrypoint = self.entrypoint;
        let mut entry_distance =
            distance2_limited(&vector, &dataset.vector_at(entrypoint), u64::MAX);

        for current_level in ((level + 1)..=self.max_level).rev() {
            (entrypoint, entry_distance) =
                self.greedy_search(dataset, &vector, entrypoint, entry_distance, current_level);
        }

        for current_level in (1..=level.min(self.max_level)).rev() {
            let candidates = self.search_layer(
                dataset,
                &vector,
                entrypoint,
                HNSW_EF_CONSTRUCTION,
                current_level,
            );
            let selected = self.select_neighbors(dataset, &candidates, HNSW_UPPER_M);
            self.set_upper_neighbors(index, current_level, &selected);

            for neighbor in selected {
                self.add_backlink(dataset, neighbor, index, current_level, HNSW_UPPER_M);
            }

            if let Some(nearest) = candidates.first() {
                entrypoint = nearest.index;
            }
        }

        let candidates = self.search_layer(dataset, &vector, entrypoint, HNSW_EF_CONSTRUCTION, 0);
        let selected = self.select_neighbors(dataset, &candidates, HNSW_M);
        self.set_level0_neighbors(index, &selected);

        for neighbor in selected {
            self.add_backlink(dataset, neighbor, index, 0, HNSW_M);
        }

        if level > self.max_level {
            self.entrypoint = index;
            self.max_level = level;
        }
    }

    fn greedy_search(
        &self,
        dataset: &ReferenceDataset,
        vector: &ReferenceVector,
        start: usize,
        start_distance: u64,
        level: u8,
    ) -> (usize, u64) {
        let mut entrypoint = start;
        let mut entry_distance = start_distance;
        let mut changed = true;

        while changed {
            changed = false;

            for neighbor in self.neighbors(entrypoint, level) {
                let distance =
                    distance2_limited(vector, &dataset.vector_at(neighbor), entry_distance);
                if distance < entry_distance {
                    entrypoint = neighbor;
                    entry_distance = distance;
                    changed = true;
                }
            }
        }

        (entrypoint, entry_distance)
    }

    fn search_layer(
        &mut self,
        dataset: &ReferenceDataset,
        vector: &ReferenceVector,
        entrypoint: usize,
        ef: usize,
        level: u8,
    ) -> Vec<NearestCandidate> {
        self.next_visit_mark();
        let entry_distance = distance2_limited(vector, &dataset.vector_at(entrypoint), u64::MAX);
        let mut candidates = BinaryHeap::new();
        let mut nearest = TopBuildNearest::new(ef);

        self.mark_visited(entrypoint);
        candidates.push(Reverse(BuildHeapCandidate {
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

            let neighbors = self.neighbors(candidate.index, level).collect::<Vec<_>>();
            for neighbor in neighbors {
                if self.is_visited(neighbor) {
                    continue;
                }

                self.mark_visited(neighbor);
                let limit = nearest.current_worst_distance();
                let distance = distance2_limited(vector, &dataset.vector_at(neighbor), limit);

                if limit == u64::MAX || distance < limit {
                    candidates.push(Reverse(BuildHeapCandidate {
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

    fn select_neighbors(
        &self,
        dataset: &ReferenceDataset,
        candidates: &[NearestCandidate],
        capacity: usize,
    ) -> Vec<usize> {
        let mut sorted = candidates.to_vec();
        sorted.sort_unstable_by_key(|candidate| (candidate.distance, candidate.index));
        let mut selected: Vec<NearestCandidate> = Vec::with_capacity(capacity);

        for candidate in &sorted {
            if selected.len() == capacity {
                break;
            }

            let candidate_vector = dataset.vector_at(candidate.index);
            let redundant = selected.iter().any(|selected| {
                distance2_limited(
                    &candidate_vector,
                    &dataset.vector_at(selected.index),
                    candidate.distance,
                ) < candidate.distance
            });

            if !redundant {
                selected.push(*candidate);
            }
        }

        for candidate in sorted {
            if selected.len() == capacity {
                break;
            }

            if !selected
                .iter()
                .any(|selected| selected.index == candidate.index)
            {
                selected.push(candidate);
            }
        }

        selected
            .into_iter()
            .map(|candidate| candidate.index)
            .collect()
    }

    fn add_backlink(
        &mut self,
        dataset: &ReferenceDataset,
        target: usize,
        source: usize,
        level: u8,
        capacity: usize,
    ) {
        let target_vector = dataset.vector_at(target);
        let mut candidates = self
            .neighbors(target, level)
            .map(|index| NearestCandidate {
                index,
                distance: distance2_limited(&target_vector, &dataset.vector_at(index), u64::MAX),
            })
            .collect::<Vec<_>>();

        if !candidates.iter().any(|candidate| candidate.index == source) {
            candidates.push(NearestCandidate {
                index: source,
                distance: distance2_limited(&target_vector, &dataset.vector_at(source), u64::MAX),
            });
        }

        let mut selected = self.select_neighbors(dataset, &candidates, capacity);
        self.keep_competitive_backlink(dataset, target, source, &mut selected, capacity);

        if level == 0 {
            self.set_level0_neighbors(target, &selected);
        } else {
            self.set_upper_neighbors(target, level, &selected);
        }
    }

    fn keep_competitive_backlink(
        &self,
        dataset: &ReferenceDataset,
        target: usize,
        source: usize,
        selected: &mut Vec<usize>,
        capacity: usize,
    ) {
        if selected.contains(&source) {
            return;
        }

        if selected.len() < capacity {
            selected.push(source);
            return;
        }

        let target_vector = dataset.vector_at(target);
        let source_distance =
            distance2_limited(&target_vector, &dataset.vector_at(source), u64::MAX);
        let Some((slot, worst_distance)) = selected
            .iter()
            .enumerate()
            .map(|(slot, neighbor)| {
                (
                    slot,
                    distance2_limited(&target_vector, &dataset.vector_at(*neighbor), u64::MAX),
                )
            })
            .max_by_key(|(_, distance)| *distance)
        else {
            return;
        };

        if source_distance < worst_distance {
            selected[slot] = source;
        }
    }

    fn neighbors(&self, index: usize, level: u8) -> impl Iterator<Item = usize> + '_ {
        let range = if level == 0 {
            NeighborRange::Level0(0..HNSW_M)
        } else if level > self.levels[index] {
            NeighborRange::Empty(0..0)
        } else {
            let start = (level as usize - 1) * HNSW_UPPER_M;
            NeighborRange::Upper(start..start + HNSW_UPPER_M)
        };

        range.filter_map(move |slot| {
            let neighbor = match slot {
                NeighborSlot::Level0(slot) => self.level0[index][slot],
                NeighborSlot::Upper(slot) => self.upper[index][slot],
            };
            (neighbor != EMPTY_NEIGHBOR).then_some(neighbor as usize)
        })
    }

    fn set_level0_neighbors(&mut self, index: usize, selected: &[usize]) {
        self.level0[index] = [EMPTY_NEIGHBOR; HNSW_M];
        for (slot, neighbor) in selected.iter().take(HNSW_M).enumerate() {
            self.level0[index][slot] = *neighbor as u32;
        }
    }

    fn set_upper_neighbors(&mut self, index: usize, level: u8, selected: &[usize]) {
        let start = (level as usize - 1) * HNSW_UPPER_M;
        let end = start + HNSW_UPPER_M;
        self.upper[index][start..end].fill(EMPTY_NEIGHBOR);

        for (slot, neighbor) in selected.iter().take(HNSW_UPPER_M).enumerate() {
            self.upper[index][start + slot] = *neighbor as u32;
        }
    }

    fn next_visit_mark(&mut self) {
        self.visit_mark = self.visit_mark.wrapping_add(1);
        if self.visit_mark == 0 {
            self.visited.fill(0);
            self.visit_mark = 1;
        }
    }

    fn mark_visited(&mut self, index: usize) {
        self.visited[index] = self.visit_mark;
    }

    fn is_visited(&self, index: usize) -> bool {
        self.visited[index] == self.visit_mark
    }
}

enum NeighborSlot {
    Level0(usize),
    Upper(usize),
}

enum NeighborRange {
    Level0(std::ops::Range<usize>),
    Upper(std::ops::Range<usize>),
    Empty(std::ops::Range<usize>),
}

impl Iterator for NeighborRange {
    type Item = NeighborSlot;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            NeighborRange::Level0(range) => range.next().map(NeighborSlot::Level0),
            NeighborRange::Upper(range) => range.next().map(NeighborSlot::Upper),
            NeighborRange::Empty(range) => range.next().map(NeighborSlot::Level0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BuildHeapCandidate {
    distance: u64,
    index: usize,
}

impl Ord for BuildHeapCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for BuildHeapCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct TopBuildNearest {
    candidates: Vec<NearestCandidate>,
    capacity: usize,
}

impl TopBuildNearest {
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

fn random_level(rng: &mut u64) -> u8 {
    let mut level = 0_u8;

    while level < u8::MAX && next_random(rng).is_multiple_of(HNSW_M as u64) {
        level += 1;
    }

    level
}

fn next_random(rng: &mut u64) -> u64 {
    let mut x = *rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *rng = x;
    x
}

fn write_vector(writer: &mut BufWriter<File>, vector: &ReferenceVector) -> Result<()> {
    for value in vector {
        writer.write_all(&value.to_le_bytes())?;
    }

    Ok(())
}
