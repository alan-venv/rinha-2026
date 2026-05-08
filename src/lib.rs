// COMMON
pub const VECTOR_DIMENSIONS: usize = 16;
pub const VECTOR_LEN: usize = VECTOR_DIMENSIONS * size_of::<i16>();
pub type ReferenceVector = [i16; VECTOR_DIMENSIONS];
pub const NEAREST_COUNT: usize = 5;

// IVF_FLAT
pub const IVF_MAGIC: &[u8; 8] = b"R26IFL04";
pub const IVF_HEADER_LEN: usize =
    IVF_MAGIC.len() + size_of::<u64>() + size_of::<u32>() + size_of::<u64>();

pub const IVF_FINE_SAMPLES: usize = 2_097_152;
pub const IVF_FINE_CENTROIDS: usize = 32768;
pub const IVF_FINE_ITERATIONS: usize = 80;

pub const IVF_COARSE_CENTROIDS: usize = 1024;
pub const IVF_COARSE_ITERATIONS: usize = 6;

pub const IVF_MAX_COARSE_PROBES: usize = 32;
pub const IVF_COARSE_PROBES: usize = 15;
pub const IVF_FINE_PROBES: usize = 12;

pub mod controller;
pub mod dto;
pub mod encoding;
pub mod memory;
pub mod service;
