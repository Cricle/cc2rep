use std::{
    io::{self, Write},
    path::PathBuf,
};

use clap::{Parser, Subcommand};
use serde_json::Value;
use tokio::net::TcpListener;
use tracing::info;

use cc2rep::{
    Settings, build_router,
    cli::{ServeConfig, parse_config_selection, prepare_serve_config, stats_endpoint},
    probe_upstream,
};

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "OpenAI Responses-compatible proxy for chat/completions backends",
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Start the proxy server
    Serve {
        /// Path to the config JSON file
        #[arg(short, long, value_name = "FILE")]
        config: Option<PathBuf>,
    },
    /// Show proxy statistics
    Stats {
        /// Proxy base URL, for example: http://127.0.0.1:8800
        #[arg(long, default_value = "http://127.0.0.1:8800", value_name = "URL")]
        url: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,cc2rep=debug".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { config } => cmd_serve(config).await,
        Commands::Stats { url } => cmd_stats(url).await,
    }
}

async fn cmd_serve(config: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let config = resolve_serve_config(config)?;
    let settings = Settings::load(&config)?;
    let addr = settings.socket_addr()?;
    info!(
        config = %config.display(),
        upstream_url = %settings.upstream_url(),
        upstream_model = %settings.upstream_model,
        "proxy configured"
    );
    let capabilities = probe_upstream(&settings).await;
    let router = build_router(settings, capabilities);
    let listener = TcpListener::bind(addr).await?;

    info!("listening on http://{}", addr);
    axum::serve(listener, router).await?;
    Ok(())
}

async fn cmd_stats(url: String) -> Result<(), Box<dyn std::error::Error>> {
    let url = stats_endpoint(&url)?;
    let client = reqwest::Client::new();
    let resp = client.get(&url).send().await?;

    if !resp.status().is_success() {
        eprintln!("error: HTTP {}", resp.status());
        let body = resp.text().await.unwrap_or_default();
        if !body.is_empty() {
            eprintln!("{body}");
        }
        std::process::exit(1);
    }

    let stats: Value = resp.json().await?;
    print_stats(&stats);
    Ok(())
}

fn resolve_serve_config(config: Option<PathBuf>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    match prepare_serve_config(config, std::path::Path::new("."))? {
        ServeConfig::Explicit(path) => Ok(path),
        ServeConfig::Candidates(candidates) => prompt_for_config(&candidates).map_err(Into::into),
    }
}

fn prompt_for_config(candidates: &[PathBuf]) -> io::Result<PathBuf> {
    println!("No --config provided. Select a config file from the current directory:");
    println!();
    for (index, path) in candidates.iter().enumerate() {
        println!("  {}. {}", index + 1, path.display());
    }
    println!();

    loop {
        print!("Enter a number [1-{}, default: 1]: ", candidates.len());
        io::stdout().flush()?;

        let mut input = String::new();
        let bytes = io::stdin().read_line(&mut input)?;
        if bytes == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "stdin closed before a config was selected",
            ));
        }

        match parse_config_selection(&input, candidates.len()) {
            Ok(index) => {
                println!("Using {}", candidates[index].display());
                println!();
                return Ok(candidates[index].clone());
            }
            Err(message) => {
                eprintln!("{message}");
            }
        }
    }
}

fn print_stats(stats: &Value) {
    let uptime = stats
        .get("uptime_seconds")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let stored = stats
        .get("stored_responses")
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let req = stats.get("requests");
    let total = req
        .and_then(|r| r.get("total"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let stream = req
        .and_then(|r| r.get("stream"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let non_stream = req
        .and_then(|r| r.get("non_stream"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completed = req
        .and_then(|r| r.get("completed"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let failed = req
        .and_then(|r| r.get("failed"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cancelled = req
        .and_then(|r| r.get("cancelled"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let inflight = req
        .and_then(|r| r.get("inflight"))
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let tok = stats.get("tokens");
    let input = tok
        .and_then(|t| t.get("input"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = tok
        .and_then(|t| t.get("output"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached = tok
        .and_then(|t| t.get("cached"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let reasoning = tok
        .and_then(|t| t.get("reasoning"))
        .and_then(Value::as_u64)
        .unwrap_or(0);

    println!("cc2rep stats (uptime: {}s)", uptime);
    println!();
    println!("Requests");
    println!("  total:      {total}");
    println!("  stream:     {stream}");
    println!("  non-stream: {non_stream}");
    println!("  completed:  {completed}");
    println!("  failed:     {failed}");
    println!("  cancelled:  {cancelled}");
    println!("  inflight:   {inflight}");
    println!();
    println!("Tokens");
    println!("  input:      {input}");
    println!("  output:     {output}");
    println!("  cached:     {cached}");
    println!("  reasoning:  {reasoning}");
    println!();
    println!("Stored responses: {stored}");
}
