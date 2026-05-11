use std::io::Result;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use actix_web::{App, HttpServer};
use mimalloc::MiMalloc;
use rinha::controller;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[actix_web::main]
async fn main() -> Result<()> {
    //let references = memory::load_references();
    //let references = Data::new(references);

    let socket = socket();
    let server = HttpServer::new(move || {
        App::new()
            //.app_data(references.clone())
            .service(controller::ready)
            .service(controller::score)
    })
    .workers(1)
    .bind_uds(&socket)?;

    permissions(&socket)?;
    server.run().await
}

fn socket() -> PathBuf {
    std::env::var("SOCKET")
        .map(PathBuf::from)
        .inspect(|p| {
            p.exists().then(|| std::fs::remove_file(p));
        })
        .inspect(|p| println!("SOCKET: {}", p.display()))
        .expect("socket path not set")
}

fn permissions(socket: &dyn AsRef<Path>) -> Result<()> {
    let permissions = std::fs::Permissions::from_mode(0o766);
    std::fs::set_permissions(socket, permissions)?;
    Ok(())
}
