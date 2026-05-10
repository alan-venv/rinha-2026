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

pub const IVF_INITIAL_PROBES: usize = 2;
pub const IVF_MAX_PROBES: usize = 8;

pub mod controller;
pub mod dto;
pub mod encoding;
pub mod memory;
pub mod service;
