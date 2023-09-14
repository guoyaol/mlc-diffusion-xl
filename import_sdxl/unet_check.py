import tvm
from web_stable_diffusion import utils
from tvm import relax
import torch
from diffusers import DiffusionPipeline
from transformers import CLIPTokenizer

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

unet = wrapper(vm["unet"], const_params_dict["unet"])


# start test

# our_inputs = torch.rand((1, 3, 64, 64)).to(torch.float32)

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



#our result
print("our result")



nd_res1 = unet(input1_nd, input2_nd, input3_nd, input4_nd, input5_nd)
our_out = nd_res1.numpy()


print(our_out)



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


import numpy as np
np.testing.assert_allclose(our_out, ref_result, atol=1e-2)