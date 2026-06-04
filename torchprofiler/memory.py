import torch

@torch.no_grad()
def measure_gpu_memory(model, input_tensor):
    if not input_tensor.is_cuda:
        return {
            "gpu_memory_mb": None
        }

    torch.cuda.reset_peak_memory_stats()

    _ = model(input_tensor)

    torch.cuda.synchronize()

    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        "gpu_memory_mb": memory_mb
    }
