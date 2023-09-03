import tvm
from web_stable_diffusion import utils
from tvm import relax
import torch

# Load the model weight parameters back.
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


import json
import numpy as np

from web_stable_diffusion import runtime


class EulerDiscreteScheduler(runtime.Scheduler):
    scheduler_name = "euler-discrete-solver"

    def __init__(self, artifact_path: str, device) -> None:
        with open(
            f"{artifact_path}/scheduler_euler_discrete_consts.json", "r"
        ) as file:
            jsoncontent = file.read()
        scheduler_consts = json.loads(jsoncontent)

        def f_convert(data, dtype):
            return [tvm.nd.array(np.array(t, dtype=dtype), device) for t in data]

        self.timesteps = f_convert(scheduler_consts["timesteps"], "int32")
        self.sigma = f_convert(scheduler_consts["sigma"], "float32")

        # self.last_model_output: tvm.nd.NDArray = tvm.nd.empty(
        #     (1, 4, 64, 64), "float32", device
        # )

    def step(
        self,
        vm: relax.VirtualMachine,
        model_output: tvm.nd.NDArray,
        sample: tvm.nd.NDArray,
        counter: int,
    ) -> tvm.nd.NDArray:
        # model_output = vm["dpm_solver_multistep_scheduler_convert_model_output"](
        #     sample, model_output, self.alpha[counter], self.sigma[counter]
        # )
        prev_latents = vm["euler_discrete_scheduler_step"](
            sample,
            model_output,
            self.sigma[counter],
            self.sigma[counter+1]
        )
        # self.last_model_output = model_output
        return prev_latents


from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer


class TVMSDPipeline:
    def __init__(
        self,
        vm: relax.VirtualMachine,
        tokenizer: CLIPTokenizer,
        tokenizer2: CLIPTokenizer,
        scheduler: runtime.Scheduler,
        tvm_device,
        param_dict,
    ):
        def wrapper(f, params):
            def wrapped_f(*args):
                return f(*args, params)

            return wrapped_f

        self.vm = vm
        self.clip_to_text_embeddings = wrapper(vm["clip"], param_dict["clip"])
        self.clip_to_text_embeddings2 = wrapper(vm["clip2"], param_dict["clip2"])
        self.unet_latents_to_noise_pred = wrapper(vm["unet"], param_dict["unet"])
        self.vae_to_image = wrapper(vm["vae"], param_dict["vae"])
        self.concat_embeddings = vm["concat_embeddings"]
        self.concat_enocder_outputs = vm["concat_enocder_outputs"]
        self.image_to_rgba = vm["image_to_rgba"]
        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer2
        self.scheduler = scheduler
        self.tvm_device = tvm_device
        self.param_dict = param_dict

    def __call__(self, prompt: str, negative_prompt: str = ""):
        # The height and width are fixed to 512.

        # Compute the embeddings for the prompt and negative prompt.
        list_text_embeddings = []

        tokenizers = [self.tokenizer, self.tokenizer2]
        text_encoders = [self.clip_to_text_embeddings, self.clip_to_text_embeddings2]

        prompt_embeds_list = []

        #prompt
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
            text_input_ids = text_inputs.input_ids.to(torch.int32)
            # Clip the text if the length exceeds the maximum allowed length.
            if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
                text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

            # Compute text embeddings.
            text_input_ids = tvm.nd.array(text_input_ids.cpu().numpy(), self.tvm_device)
            clip_output = text_encoder(text_input_ids)
            text_embeddings = clip_output[0]
            pooled_prompt_embeds = clip_output[1]

            prompt_embeds_list.append(text_embeddings)
        
        prompt_embeds = self.concat_enocder_outputs(prompt_embeds_list[0], prompt_embeds_list[1])
        print(prompt_embeds.shape)
        print(pooled_prompt_embeds.shape)

        

        #TODO: check correct, fold into TVM
        add_time_ids = torch.tensor([[1024., 1024., 0., 0., 1024., 1024.],[1024., 1024., 0., 0., 1024., 1024.]], dtype=torch.float32)
        add_time_ids = tvm.nd.array(add_time_ids, self.tvm_device)


        # Randomly initialize the latents.
        latents = torch.randn(
            (1, 4, 128, 128),
            device="cpu",
            dtype=torch.float32,
        )
        latents = tvm.nd.array(latents.numpy(), self.tvm_device)

        # UNet iteration.
        for i in tqdm(range(len(self.scheduler.timesteps))):
            #TODO: add this
            #latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            t = self.scheduler.timesteps[i]
            noise_pred = self.unet_latents_to_noise_pred(latents, t, text_embeddings, add_text_embeds, add_time_ids)
            latents = self.scheduler.step(self.vm, noise_pred, latents, i)

        # VAE decode.
        image = self.vae_to_image(latents)

        # Transform generated image to RGBA mode.
        image = self.image_to_rgba(image)
        return Image.fromarray(image.numpy().view("uint8").reshape(1024, 1024, 4))
    

pipe = TVMSDPipeline(
    vm=vm,
    tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
    tokenizer2=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
    scheduler=runtime.EulerDiscreteScheduler(artifact_path="dist", device=device),
    tvm_device=device,
    param_dict=const_params_dict,
)


import time

prompt = "Jellyfish floating in a forest"

start = time.time()
image = pipe(prompt)
end = time.time()

print(f"Time elapsed: {end - start} seconds.")