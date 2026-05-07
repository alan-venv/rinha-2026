use std::fs::File;

use anyhow::Result;
use memmap2::Mmap;
use rinha::*;

use crate::structs::{JsonLabel, ReferenceJson};

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

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn fraud_bits(&self) -> &[u8] {
        &self.fraud_bits
    }

    pub fn vector_at(&self, index: usize) -> ReferenceVector {
        self.vectors[index]
    }

    fn quantize_vector(vector: &[f32; VECTOR_DIMENSIONS]) -> ReferenceVector {
        let mut quantized = [0; VECTOR_DIMENSIONS];
        for (output, input) in quantized.iter_mut().zip(vector) {
            *output = (*input * VECTOR_SCALE).round() as i16;
        }
        quantized
    }
}
