import tvm
from web_stable_diffusion import utils
from tvm import relax
import torch
from diffusers import DiffusionPipeline
from transformers import CLIPTokenizer

from tvm.relax.frontend.torch import dynamo_capture_subgraphs
from tvm.relax.frontend.torch import from_fx
from tvm.script import relax as R

from torch import fx
from typing import Dict, List


tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# const_params_dict = utils.load_params(artifact_path="dist", device=device)
# # Load the model executable back from the shared library.
# ex = tvm.runtime.load_module("dist/stable_diffusion.so")

# vm = relax.VirtualMachine(rt_mod=ex, device=device)

def wrapper(f, params):
    def wrapped_f(*args):
        return f(*args, *params)

    return wrapped_f

# unet = wrapper(vm["unet"], const_params_dict["unet"])


########################## Handle Input ##########################

torch.manual_seed(42)
input1 = torch.rand((1, 4, 128, 128)).to(torch.float32)
input2 = torch.tensor(3).to(torch.int32)
input3 = torch.rand((2, 77, 2048)).to(torch.float32)
input4 = torch.rand((2, 1280)).to(torch.float32)
input5 = torch.rand((2, 6)).to(torch.float32)



########################## Reference Part ##########################
class UNetModelWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        # Default guidance scale factor in stable diffusion.
        self.guidance_scale = 5.0

    def forward(self, latents, timestep_tensor, text_embeddings, added_cond_kwargs_text_embeds, added_cond_kwargs_text_time_ids):
        # Latent concatenation.
        latent_model_input = torch.cat([latents] * 2, dim=0)
        # UNet forward.
        noise_pred = self.unet(latent_model_input, timestep_tensor, text_embeddings, added_cond_kwargs_text_embeds, added_cond_kwargs_text_time_ids)
        # Classifier-free guidance.
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred

device_str = "cpu"
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

ref_unet = utils.get_unet(pipe, device_str)
ref_unet.eval()
unet_to_noise_pred = UNetModelWrapper(ref_unet)

unet_to_noise_pred.eval()
with torch.no_grad():
    ref_result = unet_to_noise_pred(input1, input2, input3, input4, input5)

ref_result = ref_result.numpy()
print("ref result")
print(ref_result)


graph = fx.symbolic_trace(unet_to_noise_pred)
graph.to("cuda")
graph.eval()
input1 = input1.to("cuda")
input2 = input2.to("cuda")
input3 = input3.to("cuda")
input4 = input4.to("cuda")
input5 = input5.to("cuda")
with torch.no_grad():
    graph_out = graph(input1, input2, input3, input4, input5)
graph_out = graph_out.cpu().numpy()
print("graph out")
print(graph_out)

import numpy as np
np.testing.assert_allclose(ref_result, graph_out, rtol=1e-2, atol=1e-3)

print("model check success")

