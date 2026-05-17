use std::io::Result;

use mimalloc::MiMalloc;
use rinha::controller;
use rinha::kdtree::KdTree;
use rinha::morton::Morton;
use rinha::service::Service;
use umbral_socket::stream::UmbralServer;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() -> Result<()> {
    let morton = Morton::load_default()?;
    let kdtree = KdTree::load_default()?;
    let service = Service::new(morton, kdtree);
    let socket = socket();

    UmbralServer::new(service)
        .route(1, controller::score_handler)
        .route(2, controller::ready_handler)
        .run(&socket)
}

fn socket() -> String {
    std::env::var("SOCKET")
        .inspect(|socket| println!("SOCKET: {socket}"))
        .expect("socket path not set")
}
