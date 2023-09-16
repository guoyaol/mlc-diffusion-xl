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

target = tvm.target.Target("cuda")
device = tvm.cuda()
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

input1_nd = tvm.nd.array(input1, device=device)
input2_nd = tvm.nd.array(input2, device=device)
input3_nd = tvm.nd.array(input3, device=device)
input4_nd = tvm.nd.array(input4, device=device)
input5_nd = tvm.nd.array(input5, device=device)



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


########################## Our Part ##########################
print("our result")

def unet_latents_to_noise_pred(pipe, device_str: str) -> tvm.IRModule:

    unet = utils.get_unet(pipe, device_str)
    # unet_to_noise_pred = UNetModelWrapper(unet)
    graph = fx.symbolic_trace(unet)
    mod = from_fx(
        graph,
        [((2, 4, 128, 128), "float32"), ((), "int32"), ((2, 77, 2048), "float32"), 
         ((2, 1280), "float32"), ((2, 6), "float32")],
        keep_params_as_input=True,
    )
    return tvm.IRModule({"unet": mod["main"]})

mod = unet_latents_to_noise_pred(pipe, device_str)

mod, params = relax.frontend.detach_params(mod)
# mod = relax.pipeline.get_pipeline()(mod)
mod = relax.pipeline.transform.LegalizeOps()(mod)

target = tvm.target.Target("cuda")
device = tvm.cuda()

with target, tvm.transform.PassContext(opt_level=3):
    mod = tvm.tir.transform.DefaultGPUSchedule()(mod)

ex = relax.build(mod=mod, target=target)
vm = relax.VirtualMachine(rt_mod=ex, device=device)


def save_params(params: Dict[str, List[tvm.nd.NDArray]], artifact_path: str) -> None:
    from tvm.contrib import tvmjs

    meta_data = {}
    param_dict = {}
    for model in ["unet"]:
        meta_data[f"{model}ParamSize"] = len(params[model])
        for i, nd in enumerate(params[model]):
            param_dict[f"{model}_{i}"] = nd
    tvmjs.dump_ndarray_cache(param_dict, f"{artifact_path}/params", meta_data=meta_data)


def load_params(artifact_path: str, device) -> Dict[str, List[tvm.nd.NDArray]]:
    from tvm.contrib import tvmjs

    pdict = {}
    params, meta = tvmjs.load_ndarray_cache(f"{artifact_path}/params", device)
    for model in ["unet"]:
        plist = []
        size = meta[f"{model}ParamSize"]
        for i in range(size):
            plist.append(params[f"{model}_{i}"])
        pdict[model] = plist
    return pdict

save_params(params, artifact_path="unet_dist")

loaded_params = load_params(artifact_path="unet_dist", device=device)


imported_unet = wrapper(vm["unet"], loaded_params["unet"])
# print(vm)
# print(params.keys())



nd_res1 = imported_unet(input1_nd, input2_nd, input3_nd, input4_nd, input5_nd)
our_out = nd_res1.numpy()


print(our_out)


import numpy as np
np.testing.assert_allclose(our_out, ref_result, atol=1e-2, rtol=0.1)