from diffusers import DiffusionPipeline
import torch


import tvm
from tvm import relax
from tvm.relax.frontend.torch import dynamo_capture_subgraphs
from tvm.relax.frontend.torch import from_fx
from tvm.script import relax as R
from web_stable_diffusion import utils
from typing import Union

import torch
from torch import fx

# def step(
#     self,
#     model_output: torch.FloatTensor,
#     # timestep: Union[float, torch.FloatTensor],
#     step_index: int,
#     sample: torch.FloatTensor,
#     sigma, #  = self.sigmas[step_index]
#     next_sigma
# ):

#     # pred_original_sample = sample - sigma * model_output

#     # derivative = (sample - pred_original_sample) / sigma

#     # derivative = model_output

    
#     # dt = next_sigma - sigma

#     # prev_sample = sample + derivative * dt

#     prev_sample = sample + model_output * (next_sigma - sigma)

#     return (prev_sample,)

def euler_discrete_scheduler_steps() -> tvm.IRModule:
    bb = relax.BlockBuilder()

    # step, the function.
    sample = relax.Var("sample", R.Tensor((1, 4, 64, 64), "float32"))
    model_output = relax.Var("model_output", R.Tensor((1, 4, 64, 64), "float32"))
    sigma = relax.Var(f"sigma", R.Tensor((), "float32"))
    sigma_next = relax.Var(f"sigma", R.Tensor((), "float32"))

    with bb.function(
        "euler_discrete_scheduler_step",
        [sample, model_output, sigma, sigma_next],
    ):
        prev_sample = bb.emit(
            sample + model_output * (sigma_next - sigma),
            "prev_sample",
        )
        bb.emit_func_output(prev_sample)

    return bb.get()

def euler_discrete_scheduler_scale() -> tvm.IRModule:
    bb = relax.BlockBuilder()

    # scale, the function.
    sample = relax.Var("sample", R.Tensor((2, 4, 128, 128), "float32"))
    sigma = relax.Var(f"sigma", R.Tensor((), "float32"))

    with bb.function(
        "euler_discrete_scheduler_scale",
        [sample, sigma],
    ):
        scaled_latent_model_input = bb.emit(
            sample / ((sigma** relax.const(2.0) + relax.const(1.0)) ** relax.const(0.5)),
            "scaled_latent_model_input",
        )
        bb.emit_func_output(scaled_latent_model_input)

    return bb.get()

scheduler_step = euler_discrete_scheduler_steps()
scheduler_scale = euler_discrete_scheduler_scale()
print("successfully import")