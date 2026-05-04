// COMMON
pub const VECTOR_DIMENSIONS: usize = 14;
pub const VECTOR_SCALE: f32 = 10_000.0;
pub const VECTOR_LEN: usize = VECTOR_DIMENSIONS * size_of::<i16>();
pub type ReferenceVector = [i16; VECTOR_DIMENSIONS];

// IVF
pub const IVF_MAGIC: &[u8; 8] = b"R26IVF02";
pub const IVF_HEADER_LEN: usize =
    IVF_MAGIC.len() + size_of::<u64>() + size_of::<u32>() + size_of::<u64>();
pub const IVF_ITERATIONS: usize = 6;

pub const IVF_CENTROIDS: usize = 16384;
pub const IVF_AUX_CENTROIDS: usize = 8192;
pub const IVF_ASSIGNMENTS_PER_REFERENCE: usize = 2;
pub const IVF_INITIAL_PROBES: usize = 16;

// REFERENCE
pub const REFERENCE_MAGIC: &[u8; 8] = b"R26RFQ01";
pub const REFERENCE_HEADER_LEN: usize = REFERENCE_MAGIC.len() + size_of::<u64>();
