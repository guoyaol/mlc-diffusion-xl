from diffusers import DiffusionPipeline
import torch


import tvm
from tvm import relax
from tvm.relax.frontend.torch import dynamo_capture_subgraphs
from tvm.relax.frontend.torch import from_fx
from tvm.script import relax as R

import torch
from torch import fx

from web_stable_diffusion.utils import get_clip

print(tvm.__file__)

#TODO: support fp16
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9")

def clip_to_text_embeddings2(pipe) -> tvm.IRModule:
    # Define the wrapper torch.nn.Module for CLIP.
    class CLIPModelWrapper(torch.nn.Module):
        def __init__(self, clip):
            super().__init__()
            self.clip = clip

        def forward(self, text_input_ids):
            result = self.clip(text_input_ids, output_hidden_states=True)
            text_embeddings = result.hidden_states[-2]
            pool_text_embeddings = result.text_embeds
            return text_embeddings, pool_text_embeddings
        
    from transformers import CLIPTokenizer
    from diffusers import DiffusionPipeline

    tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    prompt = "a beautiful girl floating in galaxy"

    text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

    print("our result")
    our_out = text_inputs.input_ids

    for i in range(text_inputs.attention_mask.shape[1]):
        if text_inputs.attention_mask[0][i] == 0:
            our_out[0][i] = 0


    # input = torch.rand((1, 77)).to(torch.int32)
    input = our_out.to(torch.int32)


    clip = get_clip(pipe)
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

    return tvm.IRModule({"clip2": mod["subgraph_0"]})

clip = clip_to_text_embeddings2(pipe)
print("successful import")