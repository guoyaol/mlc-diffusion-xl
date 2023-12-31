from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
print(pipe)
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "jellyfish floating in the forest"

image = pipe(prompt=prompt, output_type="np.array").images
# image[0].save("latent.png")
print(image)
print(image.shape)

from PIL import Image
import numpy as np

# Assuming your array is named img_array
img_array = image.squeeze()  # Remove the batch dimension

print(img_array.dtype)

# Convert to uint8 if necessary
if img_array.dtype != np.uint8:
    img_array = (img_array * 255).astype(np.uint8)

# Convert the numpy array to a PIL image
pre = Image.fromarray(img_array)

# Save the image
pre.save('sd15.jpg')