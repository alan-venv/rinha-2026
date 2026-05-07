mod ivf;
mod reference;
mod structs;

use anyhow::Result;

use crate::ivf::IndexIvf;
use crate::reference::ReferenceDataset;

fn main() -> Result<()> {
    let dataset = ReferenceDataset::load()?;
    IndexIvf::build(&dataset)?;
    Ok(())
}
