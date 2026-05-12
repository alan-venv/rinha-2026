use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use memmap2::Mmap;

const MAGIC: &[u8; 8] = b"RMORTON1";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 20;
const ENTRY_LEN: usize = 45;
const DIMENSIONS: usize = 14;
const TOP_K: usize = 5;

#[derive(Clone, Copy)]
pub struct MortonEntry {
    pub key: u128,
    pub vector: [i16; DIMENSIONS],
    pub label: u8,
}

pub struct MortonIndex {
    mmap: Mmap,
    count: usize,
    window: usize,
}

impl MortonIndex {
    pub fn load_default() -> io::Result<Self> {
        Self::load("resources/index.bin")
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let input = File::open(path)?;
        let mmap = unsafe { Mmap::map(&input)? };

        if mmap.len() < HEADER_LEN || &mmap[..8] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid morton index magic",
            ));
        }

        let version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported morton index version",
            ));
        }

        let count = u64::from_le_bytes(mmap[12..20].try_into().unwrap()) as usize;
        let expected_len = HEADER_LEN + (count * ENTRY_LEN);
        if mmap.len() != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid morton index length",
            ));
        }

        Ok(Self {
            mmap,
            count,
            window: morton_window(),
        })
    }

    pub fn write(path: impl AsRef<Path>, entries: &[MortonEntry]) -> io::Result<()> {
        let mut output = BufWriter::new(File::create(path)?);
        output.write_all(MAGIC)?;
        output.write_all(&VERSION.to_le_bytes())?;
        output.write_all(&(entries.len() as u64).to_le_bytes())?;

        for entry in entries {
            output.write_all(&entry.key.to_le_bytes())?;
            for value in entry.vector {
                output.write_all(&value.to_le_bytes())?;
            }
            output.write_all(&[entry.label])?;
        }

        output.flush()
    }

    pub fn fraud_score(&self, vector: &[i16; DIMENSIONS]) -> f32 {
        if self.count == 0 {
            return 0.0;
        }

        let key = morton_key(vector);
        let position = self.lower_bound(key);
        let start = position.saturating_sub(self.window);
        let end = (position + self.window + 1).min(self.count);
        let mut best = [(u32::MAX, 0_u8); TOP_K];

        for index in start..end {
            let entry = self.entry_at(index);
            let distance = l1_distance(vector, &entry.vector);
            if distance >= best[TOP_K - 1].0 {
                continue;
            }

            let mut index = TOP_K - 1;
            while index > 0 && distance < best[index - 1].0 {
                best[index] = best[index - 1];
                index -= 1;
            }
            best[index] = (distance, entry.label);
        }

        let frauds = best.iter().filter(|(_, label)| *label == 1).count();
        frauds as f32 / TOP_K as f32
    }

    fn lower_bound(&self, key: u128) -> usize {
        let mut left = 0;
        let mut right = self.count;

        while left < right {
            let middle = left + ((right - left) / 2);
            if self.key_at(middle) < key {
                left = middle + 1;
            } else {
                right = middle;
            }
        }

        left
    }

    fn entry_at(&self, index: usize) -> MortonEntry {
        let start = HEADER_LEN + (index * ENTRY_LEN);
        decode_entry(&self.mmap[start..start + ENTRY_LEN])
    }

    fn key_at(&self, index: usize) -> u128 {
        let start = HEADER_LEN + (index * ENTRY_LEN);
        u128::from_le_bytes(self.mmap[start..start + 16].try_into().unwrap())
    }
}

pub fn entry(vector: [i16; DIMENSIONS], label: u8) -> MortonEntry {
    MortonEntry {
        key: morton_key(&vector),
        vector,
        label,
    }
}

pub fn morton_key(vector: &[i16; DIMENSIONS]) -> u128 {
    let mut key = 0_u128;

    for bit in 0..8 {
        for (dimension, value) in vector.iter().enumerate() {
            let byte = quantize_to_byte(dimension, *value);
            let source_bit = (byte >> (7 - bit)) & 1;
            key = (key << 1) | source_bit as u128;
        }
    }

    key
}

fn quantize_to_byte(dimension: usize, value: i16) -> u8 {
    match dimension {
        5 | 6 => quantize_optional(value),
        _ => quantize_normal(value),
    }
}

fn quantize_normal(value: i16) -> u8 {
    let clamped = (value as i32).clamp(0, 10_000) as u32;
    ((clamped * 255 + 5_000) / 10_000) as u8
}

fn quantize_optional(value: i16) -> u8 {
    let shifted = (value as i32 + 10_000).clamp(0, 20_000) as u32;
    ((shifted * 255 + 10_000) / 20_000) as u8
}

fn l1_distance(left: &[i16; DIMENSIONS], right: &[i16; DIMENSIONS]) -> u32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
        .sum()
}

fn decode_entry(raw: &[u8]) -> MortonEntry {
    let key = u128::from_le_bytes(raw[..16].try_into().unwrap());
    let mut vector = [0_i16; DIMENSIONS];
    let mut offset = 16;

    for value in &mut vector {
        *value = i16::from_le_bytes(raw[offset..offset + 2].try_into().unwrap());
        offset += 2;
    }

    MortonEntry {
        key,
        vector,
        label: raw[offset],
    }
}

fn morton_window() -> usize {
    std::env::var("MORTON_WINDOW")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4096)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantizes_normal_dimensions_to_full_byte_range() {
        assert_eq!(quantize_to_byte(0, -10_000), 0);
        assert_eq!(quantize_to_byte(0, 0), 0);
        assert_eq!(quantize_to_byte(0, 5_000), 128);
        assert_eq!(quantize_to_byte(0, 10_000), 255);
    }

    #[test]
    fn quantizes_optional_dimensions_with_missing_sentinel() {
        assert_eq!(quantize_to_byte(5, -10_000), 0);
        assert_eq!(quantize_to_byte(5, 0), 128);
        assert_eq!(quantize_to_byte(5, 5_000), 191);
        assert_eq!(quantize_to_byte(5, 10_000), 255);
        assert_eq!(quantize_to_byte(6, -10_000), 0);
    }

    #[test]
    fn l1_distance_uses_full_vector() {
        let left = [0; DIMENSIONS];
        let mut right = [0; DIMENSIONS];
        right[0] = 10;
        right[13] = -20;

        assert_eq!(l1_distance(&left, &right), 30);
    }
}
