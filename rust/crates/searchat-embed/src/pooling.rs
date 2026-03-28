/// Compute mean pooling over token embeddings, masked by attention_mask, then L2-normalize.
///
/// `hidden_state` is a flat slice of shape [seq_len * dim] (C-contiguous row-major).
/// `attention_mask` has length seq_len, values 0 or 1 (stored as i64).
pub fn mean_pool(
    hidden_state: &[f32],
    attention_mask: &[i64],
    seq_len: usize,
    dim: usize,
) -> Vec<f32> {
    debug_assert_eq!(hidden_state.len(), seq_len * dim);
    debug_assert_eq!(attention_mask.len(), seq_len);

    let mut pooled = vec![0.0f32; dim];
    let mut mask_sum = 0.0f32;

    for t in 0..seq_len {
        let mask = attention_mask[t] as f32;
        mask_sum += mask;
        for d in 0..dim {
            pooled[d] += hidden_state[t * dim + d] * mask;
        }
    }

    if mask_sum > 0.0 {
        for d in 0..dim {
            pooled[d] /= mask_sum;
        }
    }

    // L2 normalize
    let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for d in 0..dim {
            pooled[d] /= norm;
        }
    }

    pooled
}
