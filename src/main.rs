use std::path::PathBuf;

use clap::Parser;
use tokio::net::TcpListener;
use tracing::info;

use cc2rep::{Settings, build_router, probe_upstream};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Cli {
    #[arg(long)]
    config: PathBuf,
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
    let settings = Settings::load(&cli.config)?;
    let addr = settings.socket_addr()?;
    info!(
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
