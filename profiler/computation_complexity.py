import torch
from thop import profile as thop_profile



@torch.no_grad()
def count_complexity(model, input_tensor):
    macs, params = thop_profile(
        model,
        inputs=(input_tensor,),
        verbose=False
    )

    return {
        "macs": macs,
        "flops": 2 * macs,
        "params_from_thop": params,
    }