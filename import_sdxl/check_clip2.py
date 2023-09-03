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

# def clip_to_text_embeddings(pipe) -> tvm.IRModule:
#     # Define the wrapper torch.nn.Module for CLIP.
#     class CLIPModelWrapper(torch.nn.Module):
#         def __init__(self, clip):
#             super().__init__()
#             self.clip = clip

#         def forward(self, text_input_ids):
#             text_embeddings = self.clip(text_input_ids)[0]
#             return text_embeddings

#     clip = pipe.text_encoder_2
#     clip_to_text_embeddings = CLIPModelWrapper(clip)

#     # Create random input (77 is the maximum length).
#     text_input_ids = torch.rand((1, 77)).to(torch.int32)
#     # Capture CLIP's computational graph.
#     # mod = dynamo_capture_subgraphs(
#     #     clip_to_text_embeddings.forward,
#     #     text_input_ids,
#     #     keep_params_as_input=True,
#     # )
#     # assert len(mod.functions) == 1

#     # return tvm.IRModule({"clip": mod["subgraph_0"]})

#     graph = fx.symbolic_trace(clip_to_text_embeddings)
#     mod = from_fx(
#         graph,
#         [((1, 77), "int32")],
#         keep_params_as_input=True,
#     )
#     return tvm.IRModule({"clip2": mod["main"]})

# clip = clip_to_text_embeddings(pipe)
# print("successful import")

# print(pipe)

pipe.text_encoder_2.eval()

from web_stable_diffusion.utils import get_clip

with torch.no_grad():
    clip2 = get_clip(pipe)

    text_input_ids = torch.rand((1, 77)).to(torch.int32)


    print("our out")
    out = clip2(text_input_ids)
    print(out)
    print("text embeds")
    print(out.text_embeds.squeeze(1))
    print("last_hidden_state")
    print(out.last_hidden_state)



    print("ref out")
    ref_out = pipe.text_encoder_2(text_input_ids)
    print(ref_out)

    print("ref out 0")
    print(ref_out[0])
