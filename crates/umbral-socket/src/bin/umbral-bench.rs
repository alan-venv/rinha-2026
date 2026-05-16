use std::env;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use umbral_socket::stream::{UmbralClient, UmbralServer};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

const DEFAULT_CONCURRENCY: usize = 64;
const DEFAULT_REQUESTS: usize = 100_000;
const DEFAULT_PAYLOAD_BYTES: usize = 32;
const DEFAULT_WARMUP: usize = 1_000;
const DEFAULT_SOCKET: &str = "/tmp/umbral-bench.sock";

#[derive(Default)]
struct State;

struct Config {
    concurrency: usize,
    requests: usize,
    payload_bytes: usize,
    warmup: usize,
    socket: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            concurrency: DEFAULT_CONCURRENCY,
            requests: DEFAULT_REQUESTS,
            payload_bytes: DEFAULT_PAYLOAD_BYTES,
            warmup: DEFAULT_WARMUP,
            socket: DEFAULT_SOCKET.to_string(),
        }
    }
}

fn main() -> Result<(), BoxError> {
    let config = parse_args()?;
    config.validate()?;

    remove_socket_if_exists(&config.socket)?;

    let server_socket = config.socket.clone();
    let _server_handle = thread::spawn(move || {
        let _ = UmbralServer::new(State)
            .route(1, |_, _| Ok(b"ok"))
            .run(&server_socket);
    });

    run_benchmark(&config)
}

fn run_benchmark(config: &Config) -> Result<(), BoxError> {
    wait_for_socket(&config.socket)?;

    let payload = vec![b'x'; config.payload_bytes];
    let mut warmup_client = UmbralClient::new(&config.socket)?;
    for _ in 0..config.warmup {
        warmup_client.send(1, &payload)?;
    }

    let benchmark_start = Instant::now();
    let mut handles = Vec::with_capacity(config.concurrency);
    let socket = Arc::<str>::from(config.socket.as_str());
    let payload = Arc::new(payload);
    let per_task = config.requests / config.concurrency;
    let remainder = config.requests % config.concurrency;

    for task_index in 0..config.concurrency {
        let socket = socket.clone();
        let payload = payload.clone();
        let task_requests = per_task + usize::from(task_index < remainder);

        handles.push(thread::spawn(move || {
            let mut client = UmbralClient::new(socket.as_ref())?;
            let mut latencies = Vec::with_capacity(task_requests);
            for _ in 0..task_requests {
                let start = Instant::now();
                client.send(1, payload.as_ref())?;
                latencies.push(start.elapsed());
            }
            Ok::<_, io::Error>(latencies)
        }));
    }

    let mut latencies = Vec::with_capacity(config.requests);
    for handle in handles {
        let task_latencies = handle
            .join()
            .map_err(|_| io::Error::other("benchmark thread panicked"))??;
        latencies.extend(task_latencies);
    }

    let total_time = benchmark_start.elapsed();

    if latencies.len() != config.requests {
        return Err(io::Error::other(format!(
            "collected {} latencies, expected {}",
            latencies.len(),
            config.requests
        ))
        .into());
    }

    latencies.sort_unstable();

    let min = latencies[0];
    let p50 = percentile(&latencies, 0.50);
    let p95 = percentile(&latencies, 0.95);
    let p99 = percentile(&latencies, 0.99);
    let p999 = percentile(&latencies, 0.999);
    let max = latencies[latencies.len() - 1];
    let requests_per_sec = config.requests as f64 / total_time.as_secs_f64();

    println!("umbral-bench");
    println!("socket: {}", config.socket);
    println!("concurrency: {}", config.concurrency);
    println!("requests: {}", config.requests);
    println!("payload_bytes: {}", config.payload_bytes);
    println!();
    println!("latency:");
    println!("  min: {}", format_duration(min));
    println!("  p50: {}", format_duration(p50));
    println!("  p95: {}", format_duration(p95));
    println!("  p99: {}", format_duration(p99));
    println!("  p999: {}", format_duration(p999));
    println!("  max: {}", format_duration(max));
    println!();
    println!("throughput:");
    println!("  total_time: {}", format_duration(total_time));
    println!("  requests_per_sec: {:.0}", requests_per_sec);

    Ok(())
}

fn parse_args() -> io::Result<Config> {
    let mut config = Config::default();
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--concurrency" => config.concurrency = parse_usize_arg(&arg, args.next())?,
            "--requests" => config.requests = parse_usize_arg(&arg, args.next())?,
            "--payload-bytes" => config.payload_bytes = parse_usize_arg(&arg, args.next())?,
            "--warmup" => config.warmup = parse_usize_arg(&arg, args.next())?,
            "--socket" => {
                config.socket = args.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--socket requires a value")
                })?;
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown argument: {arg}"),
                ));
            }
        }
    }

    Ok(config)
}

fn parse_usize_arg(name: &str, value: Option<String>) -> io::Result<usize> {
    let value = value.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{name} requires a value"),
        )
    })?;

    value.parse().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{name} must be a positive integer"),
        )
    })
}

impl Config {
    fn validate(&self) -> io::Result<()> {
        if self.requests == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "requests must be greater than zero",
            ));
        }
        if self.concurrency == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "concurrency must be greater than zero",
            ));
        }
        Ok(())
    }
}

fn remove_socket_if_exists(socket: &str) -> io::Result<()> {
    match std::fs::remove_file(socket) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn wait_for_socket(socket: &str) -> io::Result<()> {
    let deadline = Instant::now() + Duration::from_secs(1);
    while Instant::now() < deadline {
        if Path::new(socket).exists() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(1));
    }

    Err(io::Error::new(
        io::ErrorKind::TimedOut,
        format!("socket did not become available: {socket}"),
    ))
}

fn percentile(sorted: &[Duration], percentile: f64) -> Duration {
    let index = ((sorted.len() as f64 - 1.0) * percentile).round() as usize;
    sorted[index]
}

fn format_duration(duration: Duration) -> String {
    if duration.as_micros() < 1_000 {
        format!("{}us", duration.as_micros())
    } else if duration.as_millis() < 1_000 {
        format!("{:.3}ms", duration.as_secs_f64() * 1_000.0)
    } else {
        format!("{:.3}s", duration.as_secs_f64())
    }
}
