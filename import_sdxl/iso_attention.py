import numpy as np
from web_stable_diffusion.models.attention_processor import Attention
import torch
import tvm
from web_stable_diffusion import utils
from tvm import relax
from diffusers import DiffusionPipeline
from transformers import CLIPTokenizer

from tvm.relax.frontend.torch import dynamo_capture_subgraphs
from tvm.relax.frontend.torch import from_fx
from tvm.script import relax as R

from torch import fx
from typing import Dict, List

dim = 640
num_attention_heads = 10
attention_head_dim = 64
dropout = 0.0
attention_bias = False
cross_attention_dim = 2048
only_cross_attention = False
upcast_attention = False

model = Attention(
    query_dim=dim,
    heads=num_attention_heads,
    dim_head=attention_head_dim,
    dropout=dropout,
    bias=attention_bias,
    cross_attention_dim=cross_attention_dim if only_cross_attention else None,
    upcast_attention=upcast_attention,
)

torch.manual_seed(42)
input1 = torch.rand((2, 4096, 640)).to(torch.float32)

print("input1 shape: ", input1)

print("referece result")
ref_out = model(
    input1,
    encoder_hidden_states=None,
    attention_mask=None,
    # **cross_attention_kwargs,
).detach()


target = tvm.target.Target("apple/m1-gpu")
device = tvm.metal()


def wrapper(f, params):
    def wrapped_f(*args):
        return f(*args, *params)

    return wrapped_f



# ########################## Torch Graph ##########################
class UNetModelWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, input1):
            # Latent concatenation.
            result = self.unet(
                input1,
                encoder_hidden_states=None,
                attention_mask=None,
                # **cross_attention_kwargs,
            )
            return result
wrap_model = UNetModelWrapper(model)
graph = fx.symbolic_trace(wrap_model)
graph_out = graph(input1).detach().numpy()

np.testing.assert_allclose(graph_out, ref_out)

print("torch graph check success")


########################## Our Part ##########################
print("our result")


def unet_latents_to_noise_pred(model) -> tvm.IRModule:

    class UNetModelWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, input1):
            # Latent concatenation.
            result = self.unet(
                input1,
                encoder_hidden_states=None,
                attention_mask=None,
                # **cross_attention_kwargs,
            )
            return result
    model = UNetModelWrapper(model)
    graph = fx.symbolic_trace(model)
    print("successful graph")
    mod = from_fx(
        graph,
        [((2, 4096, 640), "float32")],
        keep_params_as_input=True,
    )
    return tvm.IRModule({"unet": mod["main"]})


mod = unet_latents_to_noise_pred(model)

mod, params = relax.frontend.detach_params(mod)
# mod = relax.pipeline.get_pipeline()(mod)
mod = relax.pipeline.transform.LegalizeOps()(mod)


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
    tvmjs.dump_ndarray_cache(
        param_dict, f"{artifact_path}/params", meta_data=meta_data)


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


input1_nd = tvm.nd.array(input1, device=device)

nd_res1 = imported_unet(input1_nd)
our_out = nd_res1.numpy()


# print(our_out)


np.testing.assert_allclose(our_out, ref_out)

print("check success")
