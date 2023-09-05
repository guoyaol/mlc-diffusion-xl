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

clip = wrapper(vm["clip2"], const_params_dict["clip2"])


# start test
input = torch.rand((1, 77)).to(torch.int32)

target = tvm.target.Target("cuda")
device = tvm.cuda()

input_nd = tvm.nd.array(input, device=device)


#our result
print("our result")



nd_res1 = clip(input_nd)


print(nd_res1[0])
print(nd_res1[1])


#ref result
print("ref result")

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

pipe.text_encoder.eval()
with torch.no_grad():
    ref_result = pipe.text_encoder_2(input, output_hidden_states=True)

pooled_prompt_embeds = ref_result[0]
prompt_embeds = ref_result.hidden_states[-2]


print(prompt_embeds)
print(pooled_prompt_embeds)