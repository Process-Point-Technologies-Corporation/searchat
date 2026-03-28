use std::path::Path;

use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};

use crate::error::EmbedError;

pub struct OnnxSession {
    inner: Session,
}

impl OnnxSession {
    /// Load an ONNX model from `model_path`.
    pub fn load(model_path: &Path) -> Result<Self, EmbedError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::All)
            .unwrap_or_else(|e| e.recover())
            .commit_from_file(model_path)?;
        Ok(Self { inner: session })
    }

    /// Run inference for a batch.
    ///
    /// Returns the `last_hidden_state` data as a flat Vec<f32> and the shape [batch, seq_len, dim].
    pub fn run_batch(
        &mut self,
        batch: usize,
        seq_len: usize,
        input_ids: Vec<i64>,
        attention_mask: Vec<i64>,
        token_type_ids: Vec<i64>,
    ) -> Result<(Vec<f32>, [usize; 3]), EmbedError> {
        let shape = [batch, seq_len];

        let ids_tensor = Tensor::from_array((shape, input_ids))?;
        let mask_tensor = Tensor::from_array((shape, attention_mask))?;
        let type_ids_tensor = Tensor::from_array((shape, token_type_ids))?;

        let outputs = self.inner.run(ort::inputs! {
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
            "token_type_ids" => type_ids_tensor,
        })?;

        let output = &outputs["last_hidden_state"];
        let (out_shape, data) = output.try_extract_tensor::<f32>()?;

        // out_shape is [batch, seq_len, dim]
        if out_shape.len() != 3 {
            return Err(EmbedError::InvalidShape);
        }
        let out_batch = out_shape[0] as usize;
        let out_seq = out_shape[1] as usize;
        let out_dim = out_shape[2] as usize;

        Ok((data.to_vec(), [out_batch, out_seq, out_dim]))
    }
}
