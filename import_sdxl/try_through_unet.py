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
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9")

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
    input1 = torch.rand((2, 4, 128, 128)).to(torch.float32).to("mps")
    input2 = torch.tensor(3).to("mps")
    input3 = torch.rand((2, 77, 2048)).to(torch.float32).to("mps")
    input4 = torch.rand((2, 1280)).to(torch.float32).to("mps")
    input5 = torch.rand((2, 6)).to(torch.float32).to("mps")
    input_dict = {"text_embeds": input4, "time_ids": input5}
    unet = unet.to("mps")
    print("start to infer")
    out = unet(input1, input2, input3, input4, input5)
    print(out)
    # graph = fx.symbolic_trace(unet_to_noise_pred)
    # graph.to_folder("./python_graph/")
    # mod = from_fx(
    #     graph,
    #     [((2, 4, 128, 128), "float32"), ((), "int32"), ((2, 77, 2048), "float32"), 
    #      ((2, 1280), "float32"), ((2, 6), "float32")],
    #     keep_params_as_input=True,
    # )
    # return tvm.IRModule({"unet": mod["main"]})
    return 0

torch_dev_key = "cpu"
unet = unet_latents_to_noise_pred(pipe, torch_dev_key)

print("successfully run through")
