from transformers import CLIPTokenizer
from diffusers import DiffusionPipeline

tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")


prompt = "a beautiful girl floating in galaxy"

text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

print("our result")
our_out = text_inputs.input_ids

for i in range(text_inputs.attention_mask.shape[1]):
    if text_inputs.attention_mask[0][i] == 0:
        our_out[0][i] = 0

print(our_out)



pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
print("ref tokenizer")
ref1 = pipe.tokenizer(prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",)
print(ref1.input_ids)


print("ref tokenizer_2")
ref2 = pipe.tokenizer_2(prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",)
print(ref2.input_ids)


