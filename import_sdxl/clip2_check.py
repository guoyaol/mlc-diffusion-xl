import tvm
from web_stable_diffusion import utils
from tvm import relax
import torch
from diffusers import DiffusionPipeline
from transformers import CLIPTokenizer
from diffusers import DiffusionPipeline

tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

target = tvm.target.Target("cuda")
device = tvm.cuda()
const_params_dict = utils.load_params(artifact_path="dist", device=device)
# Load the model executable back from the shared library.
ex = tvm.runtime.load_module("dist/stable_diffusion.so")

vm = relax.VirtualMachine(rt_mod=ex, device=device)

def wrapper(f, params):
    def wrapped_f(*args):
        return f(*args, *params)

    return wrapped_f

clip = wrapper(vm["clip2"], const_params_dict["clip2"])


# start test
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

print(our_out)

# input = torch.rand((1, 77)).to(torch.int32)
input = our_out.to(torch.int32)

target = tvm.target.Target("cuda")
device = tvm.cuda()

input_nd = tvm.nd.array(input, device=device)


#our result
print("our result")



nd_res1 = clip(input_nd)


# print(nd_res1[0])
# print(nd_res1[1])

our_emb = nd_res1[0].numpy()
our_pool = nd_res1[1].numpy()

print(our_emb)
print(our_pool)

#ref result
print("ref result")

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

pipe.text_encoder.eval()
with torch.no_grad():
    ref_result = pipe.text_encoder_2(input, output_hidden_states=True)

pooled_prompt_embeds = ref_result[0]
prompt_embeds = ref_result.hidden_states[-2]


print(prompt_embeds)
print(pooled_prompt_embeds)

import numpy as np
np.testing.assert_allclose(our_emb, prompt_embeds, atol=1e-2)
np.testing.assert_allclose(our_pool, pooled_prompt_embeds, atol=1e-2)