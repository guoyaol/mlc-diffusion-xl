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
    bb = relax.BlockBuilder()
    latents = relax.Var("latents", R.Tensor([1, 4, 128, 128], "float32"))

    with bb.function("cat_latents", [latents]):
        res = bb.emit(
            relax.op.concat([latents, latents], axis=0)
        )
        bb.emit_func_output(res)
    return bb.get()

cat = cat_latents()

print("successfully import")