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

def vae_to_image(pipe) -> tvm.IRModule:
    class VAEModelWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            # Scale the latents so that it can be decoded by VAE.
            latents = 1 / 0.18215 * latents
            # VAE decode
            z = self.vae.post_quant_conv(latents)
            image = self.vae.decoder(z)
            # Image normalization
            image = (image / 2 + 0.5).clamp(min=0, max=1)
            image = (image.permute(0, 2, 3, 1) * 255).round()
            return image

    vae = utils.get_vae(pipe)
    vae_to_image = VAEModelWrapper(vae)

    # z = torch.rand((1, 4, 64, 64), dtype=torch.float32)
    # mod = dynamo_capture_subgraphs(
    #     vae_to_image.forward,
    #     z,
    #     keep_params_as_input=True,
    # )
    # assert len(mod.functions) == 1

    # return tvm.IRModule({"vae": mod["subgraph_0"]})
    graph = fx.symbolic_trace(vae_to_image)
    mod = from_fx(
        graph,
        [((1, 4, 64, 64), "float32")],
        keep_params_as_input=True,
    )
    return tvm.IRModule({"vae": mod["main"]})

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
vae = vae_to_image(pipe)

print("successful import")

#our random input
input = torch.rand((1, 4, 64, 64)).to(torch.float32)

target = tvm.target.Target("apple/m1-gpu")
device = tvm.metal()

input_nd = tvm.nd.array(input, device=device)


#our result
print("our result")


from tvm import meta_schedule as ms
# db = ms.database.create(work_dir="scale_db")


with target, tvm.transform.PassContext(opt_level=3):
    # clip = relax.transform.MetaScheduleApplyDatabase()(clip)
    vae = tvm.tir.transform.DefaultGPUSchedule()(vae)
ex = relax.build(vae, target= target)
vm = relax.VirtualMachine(ex, device)

nd_res1 = vm["vae"](input_nd).numpy()

print(nd_res1)
print(nd_res1.shape)


#ref result
print("ref result")

ref_result = pipe.vae.decode(input)[0].numpy()

import numpy as np
np.testing.assert_array_equal(nd_res1, ref_result)
print("test passed")