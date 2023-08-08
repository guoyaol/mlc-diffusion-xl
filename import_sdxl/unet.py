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
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

def unet_latents_to_noise_pred(pipe, device_str: str) -> tvm.IRModule:
    class UNetModelWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet
            # Default guidance scale factor in stable diffusion.
            self.guidance_scale = 7.5

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

    unet = utils.get_unet(pipe, device_str)
    unet_to_noise_pred = UNetModelWrapper(unet)
    graph = fx.symbolic_trace(unet_to_noise_pred)
    mod = from_fx(
        graph,
        [((1, 4, 128, 128), "float32"), ((), "int32"), ((2, 77, 2048), "float32"), 
         ((2, 1280), "float32"), ((2, 6), "float32")],
        keep_params_as_input=True,
    )
    return tvm.IRModule({"unet": mod["main"]})

torch_dev_key = "cpu"
unet = unet_latents_to_noise_pred(pipe, torch_dev_key)

print("successfully import")



#our random input
input1 = torch.rand((1, 4, 128, 128)).to(torch.float32)
input2 = torch.tensor(3)
input3 = torch.rand((2, 77, 2048)).to(torch.float32)
input4 = torch.rand((2, 1280)).to(torch.float32)
input5 = torch.rand((2, 6)).to(torch.float32)

target = tvm.target.Target("apple/m1-gpu")
device = tvm.metal()

input1_nd = tvm.nd.array(input1, device=device)
input2_nd = tvm.nd.array(input2, device=device)
input3_nd = tvm.nd.array(input3, device=device)
input4_nd = tvm.nd.array(input4, device=device)
input5_nd = tvm.nd.array(input5, device=device)


#our result
print("our result")


from tvm import meta_schedule as ms

with target, tvm.transform.PassContext(opt_level=3):
    # clip = relax.transform.MetaScheduleApplyDatabase()(clip)
    unet = tvm.tir.transform.DefaultGPUSchedule()(unet)
ex = relax.build(unet, target= target)
vm = relax.VirtualMachine(ex, device)

nd_res1 = vm["unet"](input1_nd, input2_nd, input3_nd, input4_nd, input5_nd).numpy()

print(nd_res1)
print(nd_res1.shape)


#ref result
print("ref result")

input1 = input1.to("mps")
input2 = input2.to("mps")
input3 = input3.to("mps")
input4 = input4.to("mps")
input5 = input5.to("mps")
input_dict = {"text_embeds": input4, "time_ids": input5}


with torch.no_grad():
    ref_result = pipe.unet(input1, input2, input3, added_cond_kwargs = input_dict)

ref_result = ref_result.cpu().numpy()

import numpy as np
np.testing.assert_array_equal(nd_res1, ref_result)
print("test passed")

