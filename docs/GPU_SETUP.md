# GPU Acceleration Setup

Searchat automatically detects and uses GPU acceleration when available.

## Quick Start

**Default installation uses CPU** (works everywhere):
```bash
pip install .
```

**To enable GPU acceleration**, install PyTorch with CUDA support BEFORE installing Searchat:

```bash
# Uninstall CPU-only PyTorch (if already installed)
pip uninstall torch torchvision

# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Then install Searchat
pip install .
```

That's it! Searchat will automatically detect and use your GPU.

---

## Platform-Specific Instructions

### Windows / Linux with NVIDIA GPU

**Requirements:**
- NVIDIA GPU (GTX 900 series or newer)
- NVIDIA drivers installed
- CUDA 11.8+ or 12.x recommended

**Installation:**
```bash
# Check if you have NVIDIA GPU
nvidia-smi

# Install PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# For older CUDA 11.8
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### macOS with Apple Silicon (M1/M2/M3)

**Requirements:**
- Mac with M1, M2, or M3 chip
- macOS 12.3+

**Installation:**
```bash
# Install PyTorch (MPS support included by default)
pip install torch torchvision

# Searchat will automatically use MPS acceleration
```

### AMD GPUs (ROCm)

**Requirements:**
- AMD GPU with ROCm support
- ROCm 5.7+ installed

**Installation:**
```bash
# Install PyTorch with ROCm 6.2
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
```

---

## Verification

After installation, verify GPU detection:

```bash
python -c "from searchat.config import Config; c = Config.load(); print(f'Device: {c.embedding.get_device()}')"
```

Expected output:
- `Device: cuda` - NVIDIA GPU detected
- `Device: mps` - Apple Silicon GPU detected
- `Device: cpu` - No GPU or CPU-only PyTorch

When you run `searchat-web` or indexing, you'll see:
```
INFO: Initializing embedding model on device: cuda
```

---

## Performance Comparison

Tested on RTX 4080 vs CPU (Intel i9):

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Index 1000 conversations | 15 min | 1-3 min | 5-15x |
| Encode 100 text chunks | 3.2s | 0.2s | 16x |
| Search 10,000 conversations | 25ms | 15ms | 1.7x |

Results vary by GPU model and batch size.

---

## Configuration

Searchat auto-detects the best device, but you can override it:

**config/settings.toml:**
```toml
[embedding]
device = "auto"   # Default: auto-detect
# device = "cuda"  # Force NVIDIA GPU
# device = "mps"   # Force Apple Silicon
# device = "cpu"   # Force CPU only
```

**Environment variable:**
```bash
export SEARCHAT_EMBEDDING_DEVICE=cuda
```

---

## Troubleshooting

### "CUDA available: False" on Windows/Linux

**Check NVIDIA drivers:**
```bash
nvidia-smi
```

If this fails, install/update NVIDIA drivers from nvidia.com.

**Check PyTorch CUDA version:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
```

If you see `+cpu` or `CUDA: None`, PyTorch was installed without CUDA support. Reinstall:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### "MPS available: False" on macOS

- Ensure you have M1/M2/M3 chip (Apple Silicon)
- Update to macOS 12.3 or later
- Reinstall PyTorch: `pip install --upgrade torch torchvision`

### Out of Memory (OOM) errors

Reduce batch size in `config/settings.toml`:
```toml
[embedding]
batch_size = 16  # Default: 32
```

### GPU detected but not used

If you see a warning that GPU is available but not in use:
1. Uninstall CPU PyTorch: `pip uninstall torch torchvision`
2. Reinstall with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
3. Restart Searchat

---

## Why PyTorch from Special Index?

PyTorch CUDA wheels are **not on PyPI** - they're hosted on PyTorch's own index at `download.pytorch.org/whl/`. This is why you need the `--index-url` flag.

The default `pip install torch` gives you CPU-only version (works everywhere, smaller download).

---

## Summary

1. **Default**: CPU-only (no setup required)
2. **GPU**: Install PyTorch with CUDA/MPS before Searchat
3. **Verification**: Check that `device: cuda/mps` appears in logs

