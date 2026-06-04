import torch
from .computation_complexity import count_complexity
from .inference_performance import measure_latency
from .model_size import count_params
from .memory import measure_gpu_memory
from .utils import format_number


def profile_model(
    model,
    input_tensor,
    device="cuda",
    warmup=50,
    repeat=300,
    ):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    input_tensor = input_tensor.to(device)

    params = count_params(model)

    complexity = count_complexity(model, input_tensor)
    latency = measure_latency(model, input_tensor, warmup, repeat)
    memory = measure_gpu_memory(model, input_tensor)

    result = {
        "params": format_number(params["params"]),
        "trainable_params": format_number(params["trainable_params"]),
        "macs": format_number(complexity["macs"]),
        "flops": format_number(complexity["flops"]),
        "latency_ms": format_number(latency["latency_ms"]),
        "fps": format_number(latency["fps"]),
        "gpu_memory_mb": format_number(memory["gpu_memory_mb"]),
    }

    return result


