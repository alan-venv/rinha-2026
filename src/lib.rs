// REFERENCE
pub const REFERENCE_MAGIC: &[u8; 8] = b"R26RFQ01";
pub const REFERENCE_HEADER_LEN: usize = REFERENCE_MAGIC.len() + size_of::<u64>();

// COMMON
pub const VECTOR_DIMENSIONS: usize = 14;
pub const VECTOR_SCALE: f32 = 10_000.0;
pub const VECTOR_LEN: usize = VECTOR_DIMENSIONS * size_of::<i16>();
pub type ReferenceVector = [i16; VECTOR_DIMENSIONS];

// IVF_PQ
pub const IVF_MAGIC: &[u8; 8] = b"R26IPQ02";
pub const IVF_HEADER_LEN: usize = IVF_MAGIC.len()
    + size_of::<u64>()
    + size_of::<u32>()
    + size_of::<u64>()
    + size_of::<u32>()
    + size_of::<u32>()
    + size_of::<u32>();

// BINS QUALITY
pub const IVF_ITERATIONS: usize = 20;
pub const IVF_CENTROIDS: usize = 2048;

pub const PQ_LAYOUT: (usize, usize) = (2, 7);
pub const PQ_CODEWORDS: usize = 16;
