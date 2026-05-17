use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use memmap2::{Advice, Mmap};

#[cfg(not(target_arch = "x86_64"))]
compile_error!("morton AVX2 path requires x86_64");

use std::arch::x86_64::{
    __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm256_abs_epi16, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_hadd_epi32, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_set1_epi16, _mm256_sub_epi16,
};

const MAGIC: &[u8; 8] = b"RMORTON1";
const VERSION: u32 = 2;
const HEADER_LEN: usize = 20;
const ENTRY_LEN: usize = 49;
const DIMENSIONS: usize = 16;
const FAST_K: usize = 11;
const FAST_D11_LIMIT: u32 = 5_000;

#[derive(Clone, Copy)]
pub struct MortonEntry {
    pub key: u128,
    pub vector: [i16; DIMENSIONS],
    pub label: u8,
}

pub struct Morton {
    mmap: Mmap,
    count: usize,
    window: usize,
}

impl Morton {
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
        mmap.advise(Advice::WillNeed)?;
        warmup(&mmap);

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

    pub fn score(&self, vector: &[i16; DIMENSIONS]) -> Option<f32> {
        let top = self.top_neighbors::<FAST_K>(vector);
        let distance = top[FAST_K - 1].0;
        if distance > FAST_D11_LIMIT {
            return None;
        }

        match top_frauds(&top) {
            0 => Some(0.0),
            FAST_K => Some(1.0),
            _ => None,
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    fn top_neighbors<const K: usize>(&self, vector: &[i16; DIMENSIONS]) -> [(u32, u8); K] {
        let mut best = [(u32::MAX, 0_u8); K];
        if self.count == 0 {
            return best;
        }

        let key = morton_key(vector);
        let position = self.lower_bound(key);
        let start = position.saturating_sub(self.window);
        let end = (position + self.window + 1).min(self.count);

        for index in start..end {
            let entry = self.entry_at(index);
            let distance = unsafe { l1_distance_avx2(vector, &entry.vector) };
            if distance >= best[K - 1].0 {
                continue;
            }

            let mut output = K - 1;
            while output > 0 && distance < best[output - 1].0 {
                best[output] = best[output - 1];
                output -= 1;
            }
            best[output] = (distance, entry.label);
        }

        best
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

    pub(crate) fn entry_at(&self, index: usize) -> MortonEntry {
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

#[inline(always)]
unsafe fn l1_distance_avx2(left: &[i16; DIMENSIONS], right: &[i16; DIMENSIONS]) -> u32 {
    let left = unsafe { _mm256_loadu_si256(left.as_ptr() as *const __m256i) };
    let right = unsafe { _mm256_loadu_si256(right.as_ptr() as *const __m256i) };
    let diff = unsafe { _mm256_abs_epi16(_mm256_sub_epi16(left, right)) };
    let pairs = unsafe { _mm256_madd_epi16(diff, _mm256_set1_epi16(1)) };
    unsafe { horizontal_sum_i32x8(pairs) as u32 }
}

#[inline(always)]
unsafe fn horizontal_sum_i32x8(values: __m256i) -> i32 {
    let sums = unsafe { _mm256_hadd_epi32(values, values) };
    let sums = unsafe { _mm256_hadd_epi32(sums, sums) };
    let low = unsafe { _mm256_castsi256_si128(sums) };
    let high = unsafe { _mm256_extracti128_si256(sums, 1) };
    unsafe { _mm_cvtsi128_si32(_mm_add_epi32(low, high)) }
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
        .unwrap_or(512)
}

fn top_frauds<const K: usize>(top: &[(u32, u8); K]) -> usize {
    top.iter().filter(|(_, label)| *label == 1).count()
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
        right[15] = -20;

        assert_eq!(unsafe { l1_distance_avx2(&left, &right) }, 30);
    }
}
