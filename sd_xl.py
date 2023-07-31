# from huggingface_hub import login
# login()

from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.to("mps")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "a beautiful girl floating in galaxy"

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
pre.save('pre.jpg')



pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0")
pipe.to("mps")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

images = pipe(prompt=prompt, image=image, output_type="np.array").images
# images[0].save("refined.jpg")

# import numpy as np
# img_array = np.array(images[0])
print(images.shape)
# print(images.shape)
img_array = images.squeeze()  # Remove the batch dimension

# Convert to uint8 if necessary
if img_array.dtype != np.uint8:
    img_array = (img_array * 255).astype(np.uint8)

# Convert the numpy array to a PIL image
pre = Image.fromarray(img_array)

# Save the image
pre.save('refine.jpg')