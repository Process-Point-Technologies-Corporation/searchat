"""Export all-MiniLM-L6-v2 to ONNX format for Rust inference.

Uses transformers + torch directly to avoid sentence-transformers import issues.

Produces:
  rust/assets/model.onnx      — transformer backbone (no pooling layer)
  rust/assets/tokenizer.json   — HuggingFace tokenizer config

Mean pooling + L2 normalization are implemented in Rust (searchat-embed crate).

Usage:
    python scripts/export_onnx.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256
OUTPUT_DIR = Path(__file__).parent.parent / "rust" / "assets"


def export():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # Dummy inputs for tracing
    batch_size = 1
    seq_len = 16
    input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    onnx_path = OUTPUT_DIR / "model.onnx"
    print(f"Exporting to {onnx_path}...")

    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        str(onnx_path),
        opset_version=17,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
    )

    # Save tokenizer.json
    tokenizer.save_pretrained(str(OUTPUT_DIR / "_tokenizer_tmp"))
    tokenizer_src = OUTPUT_DIR / "_tokenizer_tmp" / "tokenizer.json"
    tokenizer_dst = OUTPUT_DIR / "tokenizer.json"
    if tokenizer_src.exists():
        shutil.copy2(tokenizer_src, tokenizer_dst)
    shutil.rmtree(OUTPUT_DIR / "_tokenizer_tmp", ignore_errors=True)
    print(f"Saved tokenizer to {tokenizer_dst}")

    # Validate
    print("\nValidating ONNX output...")
    validate(model, tokenizer, onnx_path)

    print(f"\nExport complete:")
    print(f"  {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  {tokenizer_dst} ({tokenizer_dst.stat().st_size / 1024:.0f} KB)")


def mean_pool(hidden_state, attention_mask):
    """Mean pooling + L2 normalize (matches Rust implementation)."""
    mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    summed = torch.sum(hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    pooled = summed / counts
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


def validate(model, tokenizer, onnx_path):
    """Compare PyTorch vs ONNX output."""
    import onnxruntime as ort

    test_texts = [
        "How do I fix a segfault in the parser?",
        "explain the authentication flow",
        "src/searchat/core/unified_storage.py",
        "A" * 500,
    ]

    session = ort.InferenceSession(str(onnx_path))

    for text in test_texts:
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )

        # PyTorch reference
        with torch.no_grad():
            pt_output = model(**encoded)
        pt_emb = mean_pool(pt_output.last_hidden_state, encoded["attention_mask"])
        pt_emb = pt_emb[0].numpy()

        # ONNX
        onnx_inputs = {
            "input_ids": encoded["input_ids"].numpy().astype(np.int64),
            "attention_mask": encoded["attention_mask"].numpy().astype(np.int64),
            "token_type_ids": encoded.get(
                "token_type_ids", torch.zeros_like(encoded["input_ids"])
            ).numpy().astype(np.int64),
        }
        onnx_hidden = session.run(None, onnx_inputs)[0]

        # Manual pooling on ONNX output
        mask = onnx_inputs["attention_mask"][:, :, np.newaxis].astype(np.float32)
        pooled = np.sum(onnx_hidden * mask, axis=1) / np.clip(np.sum(mask, axis=1), 1e-9, None)
        norm = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
        onnx_emb = (pooled / norm)[0]

        cosine = np.dot(pt_emb, onnx_emb) / (np.linalg.norm(pt_emb) * np.linalg.norm(onnx_emb))
        status = "PASS" if cosine > 0.9999 else "FAIL"
        print(f"  [{status}] {repr(text[:50])}: cosine={cosine:.6f}")


if __name__ == "__main__":
    export()
