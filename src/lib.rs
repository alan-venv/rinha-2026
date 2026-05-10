// COMMON
pub const VECTOR_DIMENSIONS: usize = 14;
pub const VECTOR_LEN: usize = VECTOR_DIMENSIONS * size_of::<i16>();
pub type ReferenceVector = [i16; VECTOR_DIMENSIONS];
pub const NEAREST_COUNT: usize = 5;

// HNSW
pub const HNSW_MAGIC: &[u8; 8] = b"R26HNW01";
pub const HNSW_HEADER_LEN: usize =
    HNSW_MAGIC.len() + size_of::<u64>() + size_of::<u32>() + size_of::<u8>() + 3 + size_of::<u64>();
pub const HNSW_M: usize = 4;
pub const HNSW_UPPER_M: usize = 2;
pub const HNSW_EF_SEARCH: usize = 128;
pub const HNSW_EF_CONSTRUCTION: usize = 64;
pub const EMPTY_NEIGHBOR: u32 = u32::MAX;

pub mod controller;
pub mod dto;
pub mod encoding;
pub mod memory;
pub mod service;
