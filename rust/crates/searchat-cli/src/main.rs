use std::path::PathBuf;
use clap::Parser;

#[derive(Parser)]
#[command(name = "searchat", version, about = "Search Claude Code conversations")]
enum Cli {
    /// Start the web server
    Serve {
        /// Directory containing web assets (index.html, static/)
        #[arg(long)]
        web_dir: Option<PathBuf>,
    },
    /// Search conversations (one-shot, prints results to stdout)
    Search {
        /// Search query
        query: String,
        /// Search mode
        #[arg(long, default_value = "cross_layer")]
        mode: String,
        /// Maximum results
        #[arg(long, default_value = "10")]
        limit: usize,
    },
    /// Trigger indexing of new conversations (spawns Python subprocess)
    Index,
    /// Show database statistics
    Status,
}

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    let config = searchat_config::settings::Config::load()
        .unwrap_or_else(|e| { eprintln!("Config error: {e}"); std::process::exit(1); });

    match cli {
        Cli::Serve { web_dir } => {
            if let Err(e) = searchat_api::start_server(config, web_dir).await {
                eprintln!("Server error: {e}");
                std::process::exit(1);
            }
        }
        Cli::Search { query, mode, limit } => {
            // Open storage, create embedder, create search engine, run search, print results as JSON
            let _ = (query, mode, limit);
            eprintln!("Search not yet connected to ONNX model");
        }
        Cli::Index => {
            // Spawn: python -m searchat.batch.index_missing
            eprintln!("Index: would spawn Python subprocess");
        }
        Cli::Status => {
            // Open storage, print statistics
            let data_dir = PathBuf::from(&config.paths.search_directory).join("data");
            match searchat_storage::UnifiedStorage::open(&data_dir) {
                Ok(storage) => {
                    match storage.get_stats() {
                        Ok(stats) => {
                            let map: std::collections::HashMap<&str, serde_json::Value> = [
                                ("conversations", serde_json::Value::from(stats.conversations)),
                                ("messages", serde_json::Value::from(stats.messages)),
                                ("exchanges", serde_json::Value::from(stats.exchanges)),
                                ("verbatim_embeddings", serde_json::Value::from(stats.verbatim_embeddings)),
                                ("palace_objects", serde_json::Value::from(stats.palace_objects)),
                                ("rooms", serde_json::Value::from(stats.rooms)),
                                ("facet_embeddings", serde_json::Value::from(stats.facet_embeddings)),
                                ("hierarchical_facets", serde_json::Value::from(stats.hierarchical_facets)),
                                ("vss_available", serde_json::Value::from(stats.vss_available)),
                                ("fts_available", serde_json::Value::from(stats.fts_available)),
                            ].into_iter().collect();
                            println!("{}", serde_json::to_string_pretty(&map).unwrap());
                        }
                        Err(e) => eprintln!("Stats error: {e}"),
                    }
                }
                Err(e) => eprintln!("Storage error: {e}"),
            }
        }
    }
}
