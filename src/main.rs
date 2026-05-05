mod consts;
mod controller;
mod dto;
mod memory;
mod service;

use std::io::{Error, Result};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::Arc;

use mimalloc::MiMalloc;
use ntex::web::{App, HttpServer};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[ntex::main]
async fn main() -> Result<()> {
    let references = memory::load_references().map_err(Error::other)?;
    let references = Arc::new(references);

    let socket = socket();
    if socket.exists() {
        let _ = std::fs::remove_file(&socket);
    }

    let server = HttpServer::new(async move || {
        App::new()
            .state(references.clone())
            .service((controller::ready, controller::score))
    })
    .workers(1)
    .bind_uds(&socket)?;

    let permissions = std::fs::Permissions::from_mode(0o766);
    std::fs::set_permissions(&socket, permissions)?;

    server.run().await
}

pub fn socket() -> PathBuf {
    std::env::var("SOCKET")
        .map(PathBuf::from)
        .inspect(|p| println!("SOCKET: {}", p.display()))
        .expect("socket path not set")
}
