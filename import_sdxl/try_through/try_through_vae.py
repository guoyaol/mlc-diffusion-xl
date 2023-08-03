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


pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9")
vae = utils.get_vae(pipe)
vae = vae.to("mps")

z = torch.rand((1, 4, 64, 64), dtype=torch.float32)
z = z.to("mps")
print("start to infer")
out = vae.decode(z)
print(out)


o_vae = pipe.vae
o_vae = vae.to("mps")

o_out = o_vae.decode(z)
print(o_out)

assert torch.allclose(out, o_out)

print("successful run through")
