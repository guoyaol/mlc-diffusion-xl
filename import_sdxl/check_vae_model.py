from diffusers import DiffusionPipeline
from web_stable_diffusion import utils
import torch

device_str = "cpu"
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

vae = utils.get_vae(pipe)
vae.eval()


latents = torch.rand((1, 4, 128, 128)).to(torch.float32)


# our_unt_noise = UNetModelWrapper(unet)



with torch.no_grad():
    our_result = vae.decode(latents, return_dict=False)[0]
# print(our_result)


# ref_unet_noise = ref_UNetModelWrapper(pipe.unet)
with torch.no_grad():
    ref_result = pipe.vae.decode(latents, return_dict=False)[0]




print(our_result.shape)
print(ref_result.shape)
import numpy as np
np.testing.assert_allclose(our_result.numpy(), ref_result.numpy())
print("model check success")