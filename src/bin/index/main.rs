mod hnsw;
mod reference;
mod structs;

use anyhow::Result;

use crate::hnsw::IndexHnsw;
use crate::reference::ReferenceDataset;

fn main() -> Result<()> {
    let dataset = ReferenceDataset::load()?.compressed_pure_regions();
    IndexHnsw::build(&dataset)?;
    Ok(())
}
