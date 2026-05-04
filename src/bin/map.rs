use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::path::PathBuf;

use rinha::{REFERENCE_MAGIC, VECTOR_DIMENSIONS, VECTOR_SCALE};

use anyhow::Result;
use memmap2::Mmap;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    vector: [f32; VECTOR_DIMENSIONS],
    label: Label,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Label {
    Legit,
    Fraud,
}

fn main() -> Result<()> {
    let input = PathBuf::from("resources/references.json");
    let output = PathBuf::from("resources/references.bin");
    let total = generate_references(&input, &output)?;
    println!("wrote {total} references to {}", output.display());
    return Ok(());
}

fn generate_references(input: &Path, output: &Path) -> Result<usize> {
    let input_file = File::open(input)?;
    let mmap = unsafe { Mmap::map(&input_file) }?;
    let records: Vec<Reference> = serde_json::from_slice(&mmap)?;

    let mut writer = ReferenceWriter::create(output, records.len())?;

    for record in &records {
        writer.write_vector(&quantize_vector(&record.vector))?;
        writer.write_fraud(matches!(record.label, Label::Fraud));
    }

    writer.finish()?;
    return Ok(records.len());
}

fn quantize_vector(vector: &[f32; VECTOR_DIMENSIONS]) -> [i16; VECTOR_DIMENSIONS] {
    let mut quantized = [0; VECTOR_DIMENSIONS];

    for (output, input) in quantized.iter_mut().zip(vector) {
        *output = (*input * VECTOR_SCALE).round() as i16;
    }

    quantized
}

struct ReferenceWriter {
    writer: BufWriter<File>,
    fraud_bits: Vec<u8>,
    index: usize,
}

impl ReferenceWriter {
    fn create(output: &Path, count: usize) -> Result<Self> {
        let output_file = File::create(output)?;
        let mut writer = BufWriter::new(output_file);
        writer.write_all(REFERENCE_MAGIC)?;
        writer.write_all(&(count as u64).to_le_bytes())?;

        Ok(Self {
            writer,
            fraud_bits: vec![0; count.div_ceil(8)],
            index: 0,
        })
    }

    fn write_vector(&mut self, vector: &[i16; VECTOR_DIMENSIONS]) -> Result<()> {
        for value in vector {
            self.writer.write_all(&value.to_le_bytes())?;
        }

        Ok(())
    }

    fn write_fraud(&mut self, is_fraud: bool) {
        if is_fraud {
            self.fraud_bits[self.index / 8] |= 1 << (self.index % 8);
        }

        self.index += 1;
    }

    fn finish(mut self) -> Result<()> {
        self.writer.write_all(&self.fraud_bits)?;
        self.writer.flush()?;
        Ok(())
    }
}
