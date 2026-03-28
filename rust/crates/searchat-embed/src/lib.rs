mod error;
mod model;
mod pooling;
mod tokenizer;

use std::path::Path;

use parking_lot::Mutex;

pub use error::EmbedError;
use model::OnnxSession;
use pooling::mean_pool;
use tokenizer::TokenizerWrapper;

/// Embedding dimension produced by MiniLM-L6-v2.
pub const EMBED_DIM: usize = 384;

/// Embeds text using a locally-loaded MiniLM ONNX model.
///
/// Load once with [`Embedder::load`], then call [`Embedder::encode`] or [`Embedder::encode_batch`]
/// as many times as needed. The session is guarded by a mutex so `&self` is usable across
/// shared references (e.g. inside an `Arc`).
pub struct Embedder {
    session: Mutex<OnnxSession>,
    tokenizer: TokenizerWrapper,
    dim: usize,
}

impl Embedder {
    /// Load from a directory that contains `model.onnx` and `tokenizer.json`.
    pub fn load(model_dir: &Path) -> Result<Self, EmbedError> {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !model_path.exists() {
            return Err(EmbedError::ModelNotFound(model_path));
        }
        if !tokenizer_path.exists() {
            return Err(EmbedError::ModelNotFound(tokenizer_path));
        }

        let session = OnnxSession::load(&model_path)?;
        let tokenizer = TokenizerWrapper::load(&tokenizer_path)?;

        log::info!(
            "Loaded embedder from {} (dim={})",
            model_dir.display(),
            EMBED_DIM
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dim: EMBED_DIM,
        })
    }

    /// Embed a single text string, returning a `Vec<f32>` of length 384.
    pub fn encode(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let embeddings = self.encode_batch(&[text])?;
        embeddings
            .into_iter()
            .next()
            .ok_or(EmbedError::InvalidShape)
    }

    /// Embed a batch of strings. Returns one `Vec<f32>` of length 384 per input.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let encoded = self.tokenizer.encode_batch(texts)?;

        // All sequences in the batch share the same padded length.
        let seq_len = encoded[0].0.len();
        let batch = encoded.len();

        // Flatten inputs in row-major order: [batch * seq_len].
        let mut flat_ids = Vec::with_capacity(batch * seq_len);
        let mut flat_mask = Vec::with_capacity(batch * seq_len);
        let mut flat_type_ids = Vec::with_capacity(batch * seq_len);

        // Keep per-sample masks for pooling (pre-flattening, still i64).
        let mut masks_per_sample: Vec<Vec<i64>> = Vec::with_capacity(batch);

        for (ids, mask, type_ids) in &encoded {
            flat_ids.extend_from_slice(ids);
            flat_mask.extend_from_slice(mask);
            flat_type_ids.extend_from_slice(type_ids);
            masks_per_sample.push(mask.clone());
        }

        let (hidden_flat, shape) = self.session.lock().run_batch(
            batch,
            seq_len,
            flat_ids,
            flat_mask,
            flat_type_ids,
        )?;

        let [out_batch, out_seq, out_dim] = shape;

        if out_batch != batch || out_seq != seq_len || out_dim != self.dim {
            return Err(EmbedError::InvalidShape);
        }

        let mut result = Vec::with_capacity(batch);
        for b in 0..batch {
            let start = b * seq_len * out_dim;
            let end = start + seq_len * out_dim;
            let sample_hidden = &hidden_flat[start..end];
            let pooled = mean_pool(sample_hidden, &masks_per_sample[b], seq_len, out_dim);
            result.push(pooled);
        }

        Ok(result)
    }

    /// Returns the embedding dimension (384).
    pub fn dim(&self) -> usize {
        self.dim
    }
}
