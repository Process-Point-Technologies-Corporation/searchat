use std::path::Path;

use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::error::EmbedError;

/// Maximum sequence length for MiniLM-L6-v2.
const MAX_SEQ_LEN: usize = 256;

pub struct TokenizerWrapper {
    inner: Tokenizer,
}

impl TokenizerWrapper {
    /// Load from a `tokenizer.json` file and configure truncation + padding for batch use.
    pub fn load(tokenizer_path: &Path) -> Result<Self, EmbedError> {
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_SEQ_LEN,
                ..Default::default()
            }))
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        tokenizer.with_padding(Some(PaddingParams::default()));

        Ok(Self { inner: tokenizer })
    }

    /// Encode a single string. Returns (input_ids, attention_mask, token_type_ids) as i64 vecs.
    #[allow(dead_code)]
    pub fn encode_single(
        &self,
        text: &str,
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>), EmbedError> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;
        Ok(encoding_to_i64_triple(&encoding))
    }

    /// Encode a batch of strings. Returns a vec of (input_ids, attention_mask, token_type_ids).
    /// All sequences are padded to the same length within the batch.
    pub fn encode_batch(
        &self,
        texts: &[&str],
    ) -> Result<Vec<(Vec<i64>, Vec<i64>, Vec<i64>)>, EmbedError> {
        let encodings = self
            .inner
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;
        Ok(encodings.iter().map(encoding_to_i64_triple).collect())
    }
}

fn encoding_to_i64_triple(
    enc: &tokenizers::Encoding,
) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
    let mask: Vec<i64> = enc.get_attention_mask().iter().map(|&x| x as i64).collect();
    let type_ids: Vec<i64> = enc.get_type_ids().iter().map(|&x| x as i64).collect();
    (ids, mask, type_ids)
}
