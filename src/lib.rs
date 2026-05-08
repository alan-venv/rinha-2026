// COMMON
pub const VECTOR_DIMENSIONS: usize = 14;
pub const VECTOR_SCALE: f32 = 10_000.0;
pub const VECTOR_LEN: usize = VECTOR_DIMENSIONS * size_of::<i16>();
pub type ReferenceVector = [i16; VECTOR_DIMENSIONS];
pub const NEAREST_COUNT: usize = 5;

// IVF_FLAT
pub const IVF_MAGIC: &[u8; 8] = b"R26IFL03";
pub const IVF_HEADER_LEN: usize =
    IVF_MAGIC.len() + size_of::<u64>() + size_of::<u32>() + size_of::<u64>();

pub const IVF_ITERATIONS: usize = 80;
pub const IVF_CENTROIDS: usize = 16384;
pub const IVF_INITIAL_PROBES: usize = 6; // 8 for safe reasons
pub const IVF_SAMPLE_MULTIPLIER: usize = 64;
pub const SIMD_EARLY_DIMENSIONS: usize = 8;
pub const BOUNDARY_COARSE_GROUP_PROBES: usize = 15; // 16 for safe reasons
pub const BOUNDARY_COARSE_GROUP_FINE_PROBES: usize = 152; // 160 for safe reasons
pub const HIERARCHY_COARSE_CENTROIDS: usize = 512;
pub const HIERARCHY_COARSE_PROBES: usize = 8;
pub const HIERARCHY_COARSE_ITERATIONS: usize = 6;

pub mod consts;
pub mod controller;
pub mod dto;
pub mod encoding;
pub mod memory;
pub mod service;
