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

#TODO: support fp16
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9")

#fix torch random seed
torch.manual_seed(0)


input1 = torch.rand((2, 4, 128, 128)).to(torch.float32).to("mps")
input2 = torch.tensor(3).to("mps")
input3 = torch.rand((2, 77, 2048)).to(torch.float32).to("mps")
input4 = torch.rand((2, 1280)).to(torch.float32).to("mps")
input5 = torch.rand((2, 6)).to(torch.float32).to("mps")

input_dict = {"text_embeds": input4, "time_ids": input5}
pipe.to("mps")

out = pipe.unet(input1, input2, input3, input_dict)

#load output from out.pt
old_out = torch.load("out.pt")

assert torch.allclose(out, old_out)
