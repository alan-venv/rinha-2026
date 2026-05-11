pub mod reference;
pub mod structs;

use anyhow::Result;

use crate::reference::ReferenceDataset;

fn main() -> Result<()> {
    let _dataset = ReferenceDataset::load()?;
    // A dataset ready to build everything needed for the current strategy.
    Ok(())
}
