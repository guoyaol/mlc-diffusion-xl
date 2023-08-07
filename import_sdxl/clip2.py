from diffusers import DiffusionPipeline
import torch


import tvm
from tvm import relax
from tvm.relax.frontend.torch import dynamo_capture_subgraphs
from tvm.relax.frontend.torch import from_fx
from tvm.script import relax as R

import torch
from torch import fx

print(tvm.__file__)

#TODO: support fp16
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9")

def clip_to_text_embeddings(pipe) -> tvm.IRModule:
    # Define the wrapper torch.nn.Module for CLIP.
    class CLIPModelWrapper(torch.nn.Module):
        def __init__(self, clip):
            super().__init__()
            self.clip = clip

        def forward(self, text_input_ids):
            text_embeddings = self.clip(text_input_ids)[0]
            return text_embeddings

    clip = pipe.text_encoder_2
    clip_to_text_embeddings = CLIPModelWrapper(clip)


    graph = fx.symbolic_trace(clip_to_text_embeddings)
    mod = from_fx(
        graph,
        [((1, 77), "int32")],
        keep_params_as_input=True,
    )
    return tvm.IRModule({"clip2": mod["main"]})

clip = clip_to_text_embeddings(pipe)
print("successful import")

# print(pipe)

text_input_ids = torch.rand((1, 77)).to(torch.int32)
out = pipe.text_encoder_2(text_input_ids)[0]
print(out)