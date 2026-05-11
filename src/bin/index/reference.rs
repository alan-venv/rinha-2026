use std::collections::HashMap;
use std::fs::File;

use anyhow::Result;
use memmap2::Mmap;
use rinha::*;

use crate::structs::{JsonLabel, ReferenceJson};

const PURE_BUCKET_SIZE: i16 = 512;
const PURE_BUCKET_REPRESENTATIVES: usize = 5;

pub struct ReferenceDataset {
    pub vectors: Vec<ReferenceVector>,
    pub fraud_bits: Vec<u8>,
}

impl ReferenceDataset {
    pub fn load() -> Result<Self> {
        let input_file = File::open("resources/references/references.json")?;
        let mmap = unsafe { Mmap::map(&input_file) }?;
        let records: Vec<ReferenceJson> = serde_json::from_slice(&mmap)?;
        let mut vectors = Vec::with_capacity(records.len());
        let mut fraud_bits = vec![0_u8; records.len().div_ceil(8)];

        for (index, record) in records.iter().enumerate() {
            vectors.push(Self::quantize_vector(&record.vector));
            if matches!(record.label, JsonLabel::Fraud) {
                fraud_bits[index / 8] |= 1 << (index % 8);
            }
        }

        Ok(Self {
            vectors,
            fraud_bits,
        })
    }

    pub fn compressed_pure_regions(self) -> Self {
        let original_len = self.len();
        let mut buckets = HashMap::<BucketKey, Bucket>::new();

        for index in 0..original_len {
            buckets
                .entry(BucketKey::from_vector(&self.vectors[index]))
                .or_default()
                .push(index, self.is_fraud(index));
        }

        let mut selected = Vec::with_capacity(original_len);
        let mut mixed_buckets = 0_usize;
        let mut compressed_buckets = 0_usize;

        for bucket in buckets.values() {
            match bucket.classification() {
                BucketClassification::Mixed => {
                    mixed_buckets += 1;
                    selected.extend_from_slice(&bucket.legit);
                    selected.extend_from_slice(&bucket.fraud);
                }
                BucketClassification::Pure(indices) => {
                    if indices.len() > PURE_BUCKET_REPRESENTATIVES {
                        compressed_buckets += 1;
                    }
                    selected.extend(self.representatives(indices));
                }
            }
        }

        selected.sort_unstable();

        let mut vectors = Vec::with_capacity(selected.len());
        let mut fraud_bits = vec![0_u8; selected.len().div_ceil(8)];

        for (output_index, input_index) in selected.into_iter().enumerate() {
            vectors.push(self.vectors[input_index]);
            if self.is_fraud(input_index) {
                fraud_bits[output_index / 8] |= 1 << (output_index % 8);
            }
        }

        println!(
            "compressed pure regions: references {} -> {}, buckets={}, mixed_buckets={}, compressed_pure_buckets={}, bucket_size={}, pure_representatives={}",
            original_len,
            vectors.len(),
            buckets.len(),
            mixed_buckets,
            compressed_buckets,
            PURE_BUCKET_SIZE,
            PURE_BUCKET_REPRESENTATIVES,
        );

        Self {
            vectors,
            fraud_bits,
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn fraud_bits(&self) -> &[u8] {
        &self.fraud_bits
    }

    pub fn vector_at(&self, index: usize) -> ReferenceVector {
        self.vectors[index]
    }

    fn is_fraud(&self, index: usize) -> bool {
        let byte = self.fraud_bits[index / 8];
        byte & (1 << (index % 8)) != 0
    }

    fn representatives(&self, indices: &[usize]) -> Vec<usize> {
        if indices.len() <= PURE_BUCKET_REPRESENTATIVES {
            return indices.to_vec();
        }

        let center = self.center(indices);
        let mut nearest = Vec::with_capacity(PURE_BUCKET_REPRESENTATIVES);

        for index in indices.iter().copied() {
            add_nearest_representative(
                &mut nearest,
                RepresentativeCandidate {
                    index,
                    distance: distance_to_center(&self.vectors[index], &center),
                },
            );
        }

        nearest
            .into_iter()
            .map(|candidate| candidate.index)
            .collect()
    }

    fn center(&self, indices: &[usize]) -> [i64; VECTOR_DIMENSIONS] {
        let mut sums = [0_i64; VECTOR_DIMENSIONS];

        for index in indices {
            for (sum, value) in sums.iter_mut().zip(&self.vectors[*index]) {
                *sum += i64::from(*value);
            }
        }

        for sum in &mut sums {
            *sum /= indices.len() as i64;
        }

        sums
    }

    fn quantize_vector(vector: &[f32; 14]) -> ReferenceVector {
        let mut quantized = [0; VECTOR_DIMENSIONS];
        for (output, input) in quantized.iter_mut().zip(vector) {
            *output = (*input * 10_000.0).round() as i16;
        }
        quantized
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct BucketKey([i16; VECTOR_DIMENSIONS]);

impl BucketKey {
    fn from_vector(vector: &ReferenceVector) -> Self {
        let mut key = [0_i16; VECTOR_DIMENSIONS];

        for (output, value) in key.iter_mut().zip(vector) {
            *output = value.div_euclid(PURE_BUCKET_SIZE);
        }

        Self(key)
    }
}

#[derive(Default)]
struct Bucket {
    legit: Vec<usize>,
    fraud: Vec<usize>,
}

impl Bucket {
    fn push(&mut self, index: usize, is_fraud: bool) {
        if is_fraud {
            self.fraud.push(index);
        } else {
            self.legit.push(index);
        }
    }

    fn classification(&self) -> BucketClassification<'_> {
        if self.legit.is_empty() {
            BucketClassification::Pure(&self.fraud)
        } else if self.fraud.is_empty() {
            BucketClassification::Pure(&self.legit)
        } else {
            BucketClassification::Mixed
        }
    }
}

enum BucketClassification<'a> {
    Pure(&'a [usize]),
    Mixed,
}

#[derive(Clone, Copy)]
struct RepresentativeCandidate {
    index: usize,
    distance: u64,
}

fn add_nearest_representative(
    nearest: &mut Vec<RepresentativeCandidate>,
    candidate: RepresentativeCandidate,
) {
    if nearest.len() < PURE_BUCKET_REPRESENTATIVES {
        nearest.push(candidate);
        return;
    }

    let Some((slot, worst)) = nearest
        .iter()
        .enumerate()
        .max_by_key(|(_, candidate)| candidate.distance)
    else {
        return;
    };

    if candidate.distance < worst.distance {
        nearest[slot] = candidate;
    }
}

fn distance_to_center(vector: &ReferenceVector, center: &[i64; VECTOR_DIMENSIONS]) -> u64 {
    vector
        .iter()
        .zip(center)
        .map(|(value, center)| {
            let delta = i64::from(*value) - *center;
            (delta * delta) as u64
        })
        .sum()
}
