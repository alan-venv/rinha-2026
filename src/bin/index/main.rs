pub mod reference;
pub mod structs;

use anyhow::Result;
use rinha::morton::{self, MortonIndex};

use crate::reference::ReferenceDataset;

fn main() -> Result<()> {
    let dataset = ReferenceDataset::load()?;
    let mut entries = Vec::with_capacity(dataset.len());

    for index in 0..dataset.len() {
        entries.push(morton::entry(
            dataset.vector_at(index),
            dataset.label_at(index),
        ));
    }

    entries.sort_unstable_by_key(|entry| entry.key);
    MortonIndex::write("resources/index.bin", &entries)?;
    println!("wrote {} morton entries", entries.len());

    Ok(())
}
