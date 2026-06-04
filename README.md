# TorchProfiler

A PyTorch model profiling tool for measuring inference latency, GPU memory usage, computational complexity, and parameter counts.

## Features

- Model parameter statistics (total & trainable parameters)
- Inference latency and FPS measurement
- GPU peak memory monitoring
- Computational complexity analysis (MACs & FLOPs)

## Dependencies

```bash
pip install torch thop
```

## Quick Start

```python
results = profile_model(model, input_tensor, device="cuda")
# Returns: params, trainable_params, macs, flops, latency_ms, fps, gpu_memory_mb
```
