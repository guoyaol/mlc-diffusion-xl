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
            result = self.clip(text_input_ids, output_hidden_states=True)
            max_length=tokenizer.model_max_length
            text_embeddings = result.hidden_states[-2]
            pool_text_embeddings = result[0]
            return text_embeddings, pool_text_embeddings

    clip = pipe.text_encoder
    clip_to_text_embeddings = CLIPModelWrapper(clip)

    # Create random input (77 is the maximum length).
    text_input_ids = torch.rand((1, 77)).to(torch.int32)
    # Capture CLIP's computational graph.
    mod = dynamo_capture_subgraphs(
        clip_to_text_embeddings.forward,
        text_input_ids,
        keep_params_as_input=True,
    )
    assert len(mod.functions) == 1

    return tvm.IRModule({"clip": mod["subgraph_0"]})

clip = clip_to_text_embeddings(pipe)
print("successful import")


# #our random input
input = torch.rand((1, 77)).to(torch.int32)

# target = tvm.target.Target("cuda")
# device = tvm.cuda()

# input_nd = tvm.nd.array(input, device=device)


# #our result
# print("our result")


# from tvm import meta_schedule as ms

# with target, tvm.transform.PassContext(opt_level=3):
#     # clip = relax.transform.MetaScheduleApplyDatabase()(clip)
#     clip = tvm.tir.transform.DefaultGPUSchedule()(clip)
# ex = relax.build(clip, target= target)
# vm = relax.VirtualMachine(ex, device)

# nd_res1 = vm["clip"](input_nd).numpy()

# print(nd_res1)
# print(nd_res1.shape)


# #ref result
# print("ref result")


ref_result = pipe.text_encoder(input, output_hidden_states=True)

non_pool = ref_result.hidden_states[-2]
pool = ref_result[0]

print("non_pool", non_pool)
print("non_pool", non_pool.shape)
print("pool", pool)
print("pool", pool.shape)
