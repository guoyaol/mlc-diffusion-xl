import tvm
from web_stable_diffusion import utils
from tvm import relax
import torch
from diffusers import DiffusionPipeline

target = tvm.target.Target("cuda")
device = tvm.cuda()
const_params_dict = utils.load_params(artifact_path="dist", device=device)
# Load the model executable back from the shared library.
ex = tvm.runtime.load_module("dist/stable_diffusion.so")

vm = relax.VirtualMachine(rt_mod=ex, device=device)

def wrapper(f, params):
    def wrapped_f(*args):
        return f(*args, params)

    return wrapped_f

vae = wrapper(vm["vae"], const_params_dict["vae"])


# start test
input = torch.rand((1, 4, 128, 128)).to(torch.float32)

target = tvm.target.Target("cuda")
device = tvm.cuda()

input_nd = tvm.nd.array(input, device=device)


#our result
print("our result")



nd_res1 = vae(input_nd)

print("our result")
print(nd_res1)


#ref result
print("ref result")


class VAEModelWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        # Scale the latents so that it can be decoded by VAE.
        latents = 1 / 0.13025 * latents
        # VAE decode
        # z = self.vae.post_quant_conv(latents)
        image = self.vae.decode(latents, return_dict=False)[0]
        # Image normalization
        image = (image / 2 + 0.5).clamp(min=0, max=1)
        image = (image.permute(0, 2, 3, 1) * 255).round()
        return image

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
vae = utils.get_vae(pipe)
vae_to_image = VAEModelWrapper(vae)

pipe.text_encoder.eval()
with torch.no_grad():
    ref_result = vae_to_image(input)


print("ref result")
print(ref_result)

