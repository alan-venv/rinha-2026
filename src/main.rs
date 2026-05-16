use std::io::Result;

use mimalloc::MiMalloc;
use rinha::controller;
use rinha::kdtree::KdTree;
use rinha::morton::MortonIndex;
use rinha::service::Service;
use umbral_socket::stream::UmbralServer;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() -> Result<()> {
    let index = MortonIndex::load_default()?;
    let kdtree = KdTree::load_default()?;
    let service = Service::new(index, kdtree);
    let socket = socket();

    UmbralServer::new(service)
        .route(controller::FRAUD_SCORE_METHOD, controller::score_handler)
        .route(controller::READY_METHOD, controller::ready_handler)
        .run(&socket)
}

fn socket() -> String {
    std::env::var("SOCKET")
        .inspect(|socket| println!("SOCKET: {socket}"))
        .expect("socket path not set")
}
