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
print(text_inputs.input_ids)


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
print(ref2)


