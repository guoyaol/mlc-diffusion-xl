from web_stable_diffusion.models.attention import BasicTransformerBlock
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

inner_dim = 640
num_attention_heads = 10
attention_head_dim = 64
dropout = 0.0
cross_attention_dim = 2048
activation_fn = "geglu"
num_embeds_ada_norm = None
attention_bias = False
only_cross_attention = False
upcast_attention = False
norm_type = "layer_norm"
norm_elementwise_affine = True

model = BasicTransformerBlock(
    inner_dim,
    num_attention_heads,
    attention_head_dim,
    dropout=dropout,
    cross_attention_dim=cross_attention_dim,
    activation_fn=activation_fn,
    num_embeds_ada_norm=num_embeds_ada_norm,
    attention_bias=attention_bias,
    only_cross_attention=only_cross_attention,
    upcast_attention=upcast_attention,
    norm_type=norm_type,
    norm_elementwise_affine=norm_elementwise_affine,
)

torch.manual_seed(42)
input1 = torch.rand((2, 4096, 640)).to(torch.float32)
input2 = torch.rand((2, 77, 2048)).to(torch.float32)

print("referece result")
ref_out = model(input1, attention_mask = None, encoder_hidden_states = input2,
            timestep = None, cross_attention_kwargs = None, class_labels = None).detach()

# input_size torch.Size([2, 4096, 640])
# attention_mask None
# encoder_hidden_states torch.Size([2, 77, 2048])
# encoder_attention_mask None
# timestep None
# cross_attention_kwargs None
# class_labels None
# print(ref_out)

target = tvm.target.Target("apple/m1-gpu")
device = tvm.metal()
# const_params_dict = utils.load_params(artifact_path="dist", device=device)
# # Load the model executable back from the shared library.
# ex = tvm.runtime.load_module("dist/stable_diffusion.so")

# vm = relax.VirtualMachine(rt_mod=ex, device=device)

def wrapper(f, params):
    def wrapped_f(*args):
        return f(*args, *params)

    return wrapped_f

# input1_nd = tvm.nd.array(input1, device=device)



########################## Our Part ##########################
print("our result")

def unet_latents_to_noise_pred(model) -> tvm.IRModule:

    class UNetModelWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, input1, input2):
            # Latent concatenation.
            result = self.unet(input1, attention_mask = None, encoder_hidden_states = input2,
                           timestep = None, cross_attention_kwargs = None, class_labels = None)
            return result
    model = UNetModelWrapper(model)
    graph = fx.symbolic_trace(model)
    print("successful graph")
    mod = from_fx(
        graph,
        [((2, 4096, 640), "float32"), ((2, 77, 2048), "float32")],
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


input1_nd = tvm.nd.array(input1, device=device)
input2_nd = tvm.nd.array(input2, device=device)

nd_res1 = imported_unet(input1_nd, input2_nd)
our_out = nd_res1.numpy()


# print(our_out)


import numpy as np
np.testing.assert_allclose(our_out, ref_out, atol=1e-2, rtol=0.1)

print("check success")

