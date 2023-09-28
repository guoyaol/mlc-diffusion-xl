from diffusers import DiffusionPipeline
import torch


import tvm
from tvm import relax
from tvm.relax.frontend.torch import dynamo_capture_subgraphs
from tvm.relax.frontend.torch import from_fx
from tvm.script import relax as R
from web_stable_diffusion import utils

import torch
from torch import fx

print(tvm.__file__)

def cat_latents() -> tvm.IRModule:
    class CatLatensWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, latents):
            # Latent concatenation.
            latent_model_input = torch.cat([latents] * 2, dim=0)
            return latent_model_input

    cat_latents = CatLatensWrapper()
    graph = fx.symbolic_trace(cat_latents)
    mod = from_fx(
        graph,
        [((1, 4, 128, 128), "float32")],
        keep_params_as_input=True,
    )
    return tvm.IRModule({"cat_latents": mod["main"]})

cat = cat_latents()

print("successfully import")