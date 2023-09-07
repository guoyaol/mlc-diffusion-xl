from diffusers import DiffusionPipeline
from web_stable_diffusion import utils
import torch

device_str = "cpu"
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

unet = utils.get_unet(pipe, device_str)
unet.eval()

input1 = torch.rand((2, 4, 128, 128)).to(torch.float32)
input2 = torch.tensor(3)
input3 = torch.rand((2, 77, 2048)).to(torch.float32)
input4 = torch.rand((2, 1280)).to(torch.float32)
input5 = torch.rand((2, 6)).to(torch.float32)

print("our result")
with torch.no_grad():
    our_result = unet(input1, input2, input3, input4, input5)
print(our_result)


added_cond_kwargs = {"text_embeds": input4, "time_ids": input5}
# noise_pred = self.unet(
#     latent_model_input,
#     t,
#     encoder_hidden_states=prompt_embeds,
#     cross_attention_kwargs=cross_attention_kwargs,
#     added_cond_kwargs=added_cond_kwargs,
#     return_dict=False,
# )[0]
pipe.unet.eval()
with torch.no_grad():
    ref_result = pipe.unet(input1, input2, input3, added_cond_kwargs = added_cond_kwargs, return_dict=False)[0]

print("ref result")
print(ref_result)



print(our_result.shape)
print(ref_result.shape)
torch.allclose(our_result, ref_result)
print("model check success")