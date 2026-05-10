// COMMON
pub const VECTOR_DIMENSIONS: usize = 16;
pub const VECTOR_LEN: usize = VECTOR_DIMENSIONS * size_of::<i16>();
pub type ReferenceVector = [i16; VECTOR_DIMENSIONS];
pub const NEAREST_COUNT: usize = 5;

// IVF_FLAT
pub const IVF_MAGIC: &[u8; 8] = b"R26IFL04";
pub const IVF_HEADER_LEN: usize =
    IVF_MAGIC.len() + size_of::<u64>() + size_of::<u32>() + size_of::<u64>();

pub const IVF_FINE_SAMPLES: usize = 3_000_000;
pub const IVF_FINE_CENTROIDS: usize = 4096;
pub const IVF_FINE_ITERATIONS: usize = 200;

pub const IVF_COARSE_CENTROIDS: usize = 64; // ~64 fine centroids per coarse
pub const IVF_COARSE_ITERATIONS: usize = 8;

pub const IVF_MAX_COARSE_PROBES: usize = 48; // coarse probes = fine centroids
pub const IVF_COARSE_PROBES: usize = 32;
pub const IVF_FINE_PROBES: usize = 2;

pub mod controller;
pub mod dto;
pub mod encoding;
pub mod memory;
pub mod service;
