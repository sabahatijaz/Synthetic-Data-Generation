# Synthetic-Data-Generation
Synthetic Data Generation with AUTOMATIC1111's API

Synthetic Data Generation is a powerful Python module designed to create variations of images while preserving specific regions of interest defined by the user. Leveraging AUTOMATIC1111's API, this module enables users to generate diverse images by keeping a selected region unchanged, allowing for versatile use cases such as object recognition, data augmentation, and training deep learning models.
Features

    Region of Interest Preservation: Define a region of interest within an image, and the module ensures this area remains consistent while the rest of the image undergoes transformations.

    Automatic Image Variation: Generate multiple variations of an image, including changes in background, clothing, and context, while maintaining the specified region of interest.

Installation

Before using the module, make sure you have Python installed on your system. Install the required dependencies using the following command:

bash

pip install requests pillow

Usage

    Import the Module:

    python

from synthetic_data_generation import predict

Specify Inputs:

    image: Input image containing the region of interest.
    img2img_prompt: Prompt for image-to-image transformation.
    word_mask: Object mask defining the region of interest.
    api_url: URL endpoint for AUTOMATIC1111's API.

Perform Image Transformation:

python

    prediction_result = predict(image, img2img_prompt, word_mask, api_url)

    The prediction_result variable will contain the generated image with preserved region of interest.

Example Usage

python

from PIL import Image
from synthetic_data_generation import predict

# Load input image
input_image = Image.open("input_image.jpg")

# Define region of interest mask
word_mask = "shoes"

# Set API endpoint URL
api_url = 'https://api.automatic1111.com/sdapi/v1/img2img'

# Define transformation prompt
img2img_prompt = "Generate diverse images with preserved shoes."

# Perform image transformation
generated_image = predict(input_image, img2img_prompt, word_mask, api_url)

# Save generated image
generated_image.save("generated_image.jpg")
