# Umbral Socket

Bytes server and client over Unix sockets.

Umbral Socket uses a binary framed protocol over Unix stream sockets. Methods
are identified by `u8`, and request/response payloads are byte slices on the
hot path.

## Installation
```bash
cargo add umbral-socket
```

## How to Use

Below are basic examples for the server and client.

### Server
Example of how to start a server that receives data and returns a fixed response.

```rust
use std::{
    io::Result,
    sync::{Arc, Mutex},
};

use umbral_socket::stream::{UmbralResponse, UmbralServer};

#[derive(Clone, Default)]
struct State {
    contents: Arc<Mutex<Vec<Vec<u8>>>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let state = State::default();
    let socket = "/tmp/umbral.sock";
    UmbralServer::new(state)
        .route(1, |state, content, _| {
            println!("CLIENT REQUEST: {}", String::from_utf8_lossy(content));
            state.contents.lock().unwrap().push(content.to_vec());
            Ok(UmbralResponse::Static(b"OK"))
        })
        .run(socket)
        .await
}
```

### Client
Example of how a client can send data using the low-allocation callback API.

```rust
use umbral_socket::stream::UmbralClient;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let socket = "/tmp/umbral.sock";
    let connections = 8;
    let client = UmbralClient::new(socket, connections).await?;

    let content = b"{\"user\":\"alan\"}";
    client
        .send_with(1, content, |response| {
            println!("SERVER RESPONSE: {}", String::from_utf8_lossy(response));
            Ok(())
        })
        .await?;

    Ok(())
}
```

`send_with` and `send_raw_with` reuse one response buffer per connection slot.
`send` and `send_raw` remain available as convenience APIs that return owned
`Bytes`.

### Exclusive Connection
For latency-sensitive dispatchers that already own the concurrency model, use
`UmbralConnection`. It wraps one Unix stream, requires `&mut self`, and does not
use the client pool, slot mutexes, or per-operation timeouts.

```rust
use umbral_socket::stream::UmbralConnection;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let mut connection = UmbralConnection::connect("/tmp/umbral.sock").await?;

    connection
        .send_with(1, b"{\"user\":\"alan\"}", |response| {
            println!("SERVER RESPONSE: {}", String::from_utf8_lossy(response));
            Ok(())
        })
        .await
}
```

## Benchmark

```bash
cargo run --release --bin umbral-bench -- \
  --connections 4 \
  --concurrency 64 \
  --requests 100000 \
  --payload-bytes 32
```

The benchmark uses `send_with`, so it measures the callback hot path without an
owned response allocation.
