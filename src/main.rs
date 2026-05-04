mod consts;
mod controller;
mod dto;
mod memory;
mod service;

use std::io::{Error, Result};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use actix_web::web::Data;
use actix_web::{App, HttpServer};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[actix_web::main]
async fn main() -> Result<()> {
    let references = memory::load_references().map_err(Error::other)?;
    let references = Data::new(references);

    let socket = socket();
    if socket.exists() {
        let _ = std::fs::remove_file(&socket);
    }

    let server = HttpServer::new(move || {
        App::new()
            .app_data(references.clone())
            .service(controller::ready)
            .service(controller::score)
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
