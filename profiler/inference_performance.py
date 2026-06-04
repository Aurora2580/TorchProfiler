import time
import torch


@torch.no_grad()
def measure_latency(model, input_tensor, warmup=50, repeat=300):

    for _ in range(warmup):
        _ = model(input_tensor)

    if input_tensor.is_cuda:
        torch.cuda.synchronize()

    start = time.time()

    for _ in range(repeat):
        _ = model(input_tensor)

    if input_tensor.is_cuda:
        torch.cuda.synchronize()

    end = time.time()

    latency_ms = (end - start) / repeat * 1000
    fps = 1000 / latency_ms


    return {
        "latency_ms": latency_ms,
        "fps": fps,
    }
