use std::fs::File;

use anyhow::Result;
use memmap2::Mmap;

use crate::structs::{JsonLabel, ReferenceJson};

pub struct ReferenceDataset {
    pub vectors: Vec<[i16; 14]>,
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

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn vector_at(&self, index: usize) -> [i16; 14] {
        self.vectors[index]
    }

    pub fn label_at(&self, index: usize) -> u8 {
        (self.fraud_bits[index / 8] >> (index % 8)) & 1
    }

    fn quantize_vector(vector: &[f32; 14]) -> [i16; 14] {
        let mut quantized = [0; 14];
        for (output, input) in quantized[..14].iter_mut().zip(vector) {
            *output = (*input * 10_000.0).round() as i16;
        }
        quantized
    }
}
