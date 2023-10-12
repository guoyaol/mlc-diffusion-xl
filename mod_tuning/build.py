from diffusers import DiffusionPipeline
import torch


import tvm
from tvm import relax
from tvm.relax.frontend.torch import dynamo_capture_subgraphs
from tvm.relax.frontend.torch import from_fx
from tvm.script import relax as R

import torch
from torch import fx

from web_stable_diffusion import utils

def clip_to_text_embeddings(pipe) -> tvm.IRModule:
    # Define the wrapper torch.nn.Module for CLIP.
    class CLIPModelWrapper(torch.nn.Module):
        def __init__(self, clip):
            super().__init__()
            self.clip = clip

        def forward(self, text_input_ids):
            result = self.clip(text_input_ids, output_hidden_states=True)
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

    clip = utils.get_clip(pipe)
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

def cat_latents() -> tvm.IRModule:
    bb = relax.BlockBuilder()
    latents = relax.Var("latents", R.Tensor([1, 4, 128, 128], "float32"))

    with bb.function("cat_latents", [latents]):
        res = bb.emit(
            relax.op.concat([latents, latents], axis=0)
        )
        bb.emit_func_output(res)
    return bb.get()

def unet_latents_to_noise_pred(pipe, device_str: str) -> tvm.IRModule:
    class UNetModelWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet
            # Default guidance scale factor in stable diffusion.
            self.guidance_scale = 5.0

        def forward(self, latents, timestep_tensor, text_embeddings, added_cond_kwargs_text_embeds, added_cond_kwargs_text_time_ids):
            # UNet forward.
            noise_pred = self.unet(latents, timestep_tensor, text_embeddings, added_cond_kwargs_text_embeds, added_cond_kwargs_text_time_ids)
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
        [((2, 4, 128, 128), "float32"), ((), "int32"), ((2, 77, 2048), "float32"), 
         ((2, 1280), "float32"), ((2, 6), "float32")],
        keep_params_as_input=True,
    )
    return tvm.IRModule({"unet": mod["main"]})


def unet_latents_to_noise_pred(pipe, device_str: str) -> tvm.IRModule:
    class UNetModelWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet
            # Default guidance scale factor in stable diffusion.
            self.guidance_scale = 5.0

        def forward(self, latents, timestep_tensor, text_embeddings, added_cond_kwargs_text_embeds, added_cond_kwargs_text_time_ids):
            # UNet forward.
            noise_pred = self.unet(latents, timestep_tensor, text_embeddings, added_cond_kwargs_text_embeds, added_cond_kwargs_text_time_ids)
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
        [((2, 4, 128, 128), "float32"), ((), "int32"), ((2, 77, 2048), "float32"), 
         ((2, 1280), "float32"), ((2, 6), "float32")],
        keep_params_as_input=True,
    )
    return tvm.IRModule({"unet": mod["main"]})

def vae_to_image(pipe) -> tvm.IRModule:
    class VAEModelWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            # Scale the latents so that it can be decoded by VAE.
            latents = 1 / 0.13025 * latents
            # VAE decode
            z = self.vae.post_quant_conv(latents)
            image = self.vae.decoder(z)
            # Image normalization
            image = (image / 2 + 0.5).clamp(min=0, max=1)
            image = (image.permute(0, 2, 3, 1) * 255).round()
            return image

    vae = utils.get_vae(pipe)
    vae_to_image = VAEModelWrapper(vae)

    graph = fx.symbolic_trace(vae_to_image)
    mod = from_fx(
        graph,
        [((1, 4, 128, 128), "float32")],
        keep_params_as_input=True,
    )
    return tvm.IRModule({"vae": mod["main"]})

def concat_embeddings() -> tvm.IRModule:
    bb = relax.BlockBuilder()
    cond_embeddings = relax.Var("cond_embeddings", R.Tensor([1, 77, 2048], "float32"))
    uncond_embeddings = relax.Var(
        "uncond_embeddings", R.Tensor([1, 77, 2048], "float32")
    )
    with bb.function("concat_embeddings", [cond_embeddings, uncond_embeddings]):
        res = bb.emit(
            relax.op.concat([cond_embeddings, uncond_embeddings], axis=0)
        )
        bb.emit_func_output(res)
    return bb.get()

def concat_enocder_outputs() -> tvm.IRModule:
    bb = relax.BlockBuilder()
    cond_embeddings = relax.Var("cond_embeddings", R.Tensor([1, 77, 768], "float32"))
    uncond_embeddings = relax.Var(
        "uncond_embeddings", R.Tensor([1, 77, 1280], "float32")
    )
    with bb.function("concat_enocder_outputs", [cond_embeddings, uncond_embeddings]):
        res = bb.emit(
            relax.op.concat([cond_embeddings, uncond_embeddings], axis=-1)
        )
        bb.emit_func_output(res)
    return bb.get()

def concat_pool_embeddings() -> tvm.IRModule:
    bb = relax.BlockBuilder()
    cond_embeddings = relax.Var("cond_embeddings", R.Tensor([1, 1280], "float32"))
    uncond_embeddings = relax.Var(
        "uncond_embeddings", R.Tensor([1, 1280], "float32")
    )
    with bb.function("concat_pool_embeddings", [cond_embeddings, uncond_embeddings]):
        res = bb.emit(
            relax.op.concat([cond_embeddings, uncond_embeddings], axis=0)
        )
        bb.emit_func_output(res)
    return bb.get()

def image_to_rgba() -> tvm.IRModule:
    from tvm import te

    def f_image_to_rgba(A):
        def fcompute(y, x):
            return (
                A[0, y, x, 0].astype("uint32")
                | (A[0, y, x, 1].astype("uint32") << 8)
                | (A[0, y, x, 2].astype("uint32") << 16)
                | tvm.tir.const(255 << 24, "uint32")
            )

        return te.compute((1024, 1024), fcompute, name="image_to_rgba")

    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor([1, 1024, 1024, 3], "float32"))
    with bb.function("image_to_rgba", [x]):
        image = bb.emit(
            bb.call_te(f_image_to_rgba, x, primfunc_name_hint="tir_image_to_rgba")
        )
        bb.emit_func_output(image)
    return bb.get()

def image_to_rgba() -> tvm.IRModule:
    from tvm import te

    def f_image_to_rgba(A):
        def fcompute(y, x):
            return (
                A[0, y, x, 0].astype("uint32")
                | (A[0, y, x, 1].astype("uint32") << 8)
                | (A[0, y, x, 2].astype("uint32") << 16)
                | tvm.tir.const(255 << 24, "uint32")
            )

        return te.compute((1024, 1024), fcompute, name="image_to_rgba")

    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor([1, 1024, 1024, 3], "float32"))
    with bb.function("image_to_rgba", [x]):
        image = bb.emit(
            bb.call_te(f_image_to_rgba, x, primfunc_name_hint="tir_image_to_rgba")
        )
        bb.emit_func_output(image)
    return bb.get()

def euler_discrete_scheduler_steps() -> tvm.IRModule:
    bb = relax.BlockBuilder()

    # step, the function.
    sample = relax.Var("sample", R.Tensor((1, 4, 128, 128), "float32"))
    model_output = relax.Var("model_output", R.Tensor((1, 4, 128, 128), "float32"))
    sigma = relax.Var(f"sigma", R.Tensor((), "float32"))
    sigma_next = relax.Var(f"sigma", R.Tensor((), "float32"))

    with bb.function(
        "euler_discrete_scheduler_step",
        [sample, model_output, sigma, sigma_next],
    ):
        prev_sample = bb.emit(
            sample + model_output * (sigma_next - sigma),
            "prev_sample",
        )
        bb.emit_func_output(prev_sample)

    return bb.get()

def euler_discrete_scheduler_scale() -> tvm.IRModule:
    bb = relax.BlockBuilder()

    # scale, the function.
    sample = relax.Var("sample", R.Tensor((2, 4, 128, 128), "float32"))
    sigma = relax.Var(f"sigma", R.Tensor((), "float32"))

    with bb.function(
        "euler_discrete_scheduler_scale",
        [sample, sigma],
    ):
        scaled_latent_model_input = bb.emit(
            sample / ((sigma** relax.const(2.0) + relax.const(1.0)) ** relax.const(0.5)),
            "scaled_latent_model_input",
        )
        bb.emit_func_output(scaled_latent_model_input)

    return bb.get()

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

torch_dev_key = "cpu"

clip = clip_to_text_embeddings(pipe)
clip2 = clip_to_text_embeddings2(pipe)
unet = unet_latents_to_noise_pred(pipe, torch_dev_key)
vae = vae_to_image(pipe)
concat_embeddings = concat_embeddings()
concat_pool_embeddings = concat_pool_embeddings()
concat_enocder_outputs = concat_enocder_outputs()
image_to_rgba = image_to_rgba()
scheduler_step = euler_discrete_scheduler_steps()
scheduler_scale = euler_discrete_scheduler_scale()
cat_latents = cat_latents()

mod: tvm.IRModule = utils.merge_irmodules(
    clip,
    clip2,
    unet,
    cat_latents,
    vae,
    concat_embeddings,
    concat_pool_embeddings,
    concat_enocder_outputs,
    image_to_rgba,
    scheduler_step,
    scheduler_scale,
)

#optimization
mod, params = relax.frontend.detach_params(mod)

mod = relax.pipeline.get_pipeline()(mod)


model_names = ["clip", "clip2", "unet", "vae"]
scheduler_func_names = ["euler_discrete_scheduler_step", "euler_discrete_scheduler_scale"]
entry_funcs = (
    model_names + scheduler_func_names  + ["image_to_rgba", "concat_embeddings", "concat_enocder_outputs", "concat_pool_embeddings", "cat_latents"]
)

# Clean up unused parts of the IRModule.
mod = relax.transform.DeadCodeElimination(entry_funcs)(mod)


mod = relax.transform.LiftTransformParams()(mod)

mod_transform, mod_deploy = utils.split_transform_deploy_mod(
    mod, model_names, entry_funcs
)


def print_relax_funcnames(mod: tvm.IRModule):
    for global_var, func in mod.functions.items():
        if isinstance(func, relax.Function):
            print(global_var.name_hint)
    print()
    
print("In IRModule for build stage:")
print_relax_funcnames(mod_transform)

print("In IRModule for deployment stage:")
print_relax_funcnames(mod_deploy)


new_params = utils.transform_params(mod_transform, params)
utils.save_params(new_params, artifact_path="dist")



import pickle

with open("deploy_mod.pkl", "wb") as f:
    pickle.dump(mod_deploy, f)

from tvm import meta_schedule as ms

target = tvm.target.Target("apple/m2-gpu")
device = tvm.metal()

with target, tvm.transform.PassContext(opt_level=3):
    mod_deploy = tvm.tir.transform.DefaultGPUSchedule()(mod_deploy)


ex = relax.build(mod=mod_deploy, target=target)

ex.export_library("dist/stable_diffusion.so")