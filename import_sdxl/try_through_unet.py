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
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9")
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9")



unet = utils.get_unet(pipe, "cpu")
input1 = torch.rand((2, 4, 128, 128)).to(torch.float32).to("mps")
input2 = torch.tensor(3).to(torch.float32).to("mps")
input3 = torch.rand((2, 77, 2048)).to(torch.float32).to("mps")
input4 = torch.rand((2, 1280)).to(torch.float32).to("mps")
input5 = torch.rand((2, 6)).to(torch.float32).to("mps")

input_dict = {"text_embeds": input4, "time_ids": input5}
unet = unet.to("mps")

out = unet(input1, input2, input3, input4, input5)

del unet

print("start second inference")

old_unet = pipe.unet
old_unet = old_unet.to("mps")
old_out = old_unet(input1, input2, input3, added_cond_kwargs = input_dict)

assert torch.allclose(out, old_out)

print("successfully run through")
