import requests
import base64
import io
from PIL import Image
from torch import autocast
from io import BytesIO
from generate_mask import prediction


def b64encode(x: bytes) -> str:
    """
    Encode bytes into Base64 representation.

    Args:
        x (bytes): Input bytes data.

    Returns:
        str: Base64-encoded string.
    """
    return base64.b64encode(x).decode("utf-8")


def predict( image, img2img_prompt, word_mask, url):
    """
    Perform image transformation and prediction.

    Args:
        radio (str): Transformation mode selection.
        dict (dict): Dictionary containing image and mask data.
        img2img_prompt (str): Prompt for image-to-image transformation.
        word_mask (str): Object mask for prediction.
        url (str): API endpoint URL.

    Returns:
        Image: Transformed or predicted image.
    """
    init_img_inpaint = image.convert("RGB").resize((512, 512))
    # init_img_inpaint.save("InpainINp.jpg")
    init_mask_inpaint = prediction(init_img_inpaint, word_mask)
    init_mask_inpaint = Image.fromarray(init_mask_inpaint)
    # init_mask_inpaint.save("desired_mask.jpg")


    image =init_img_inpaint #Image.open("InpainINp.jpg")
    base64_image = encode_image(image)

    mask =init_mask_inpaint# Image.open("desired_mask.jpg")
    base64_mask = encode_image(mask)

    data = {
        'prompt': img2img_prompt,
        'negative_prompt': "",
        'prompt_styles': [],
        'init_images': [base64_image],
        "mask": base64_mask,
        'width': 512,
        'height': 512,
        'steps': 20,
        'cfg_scale': 7,
        'denoising_strength': 0.75,
        "sampler_index": "Euler a",
        'mask_blur': 2,
        'inpainting_fill': 1,
        "batch_size": 1,
        "inpainting_mask_invert": 1,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 32,
    }
    response = requests.post(url, data=json.dumps(data), timeout=None)

    if response.status_code == 200:
        json_data = response.json()
        output = base64.b64decode(json_data['images'][0])
        output = Image.open(BytesIO(output))
        output.save('result.jpg')
        return output
    else:
        print("Failed to fetch data from the API.")


def encode_image(image):
    """Encode an image as Base64."""
    image_bytesio = io.BytesIO()
    image.save(image_bytesio, format='PNG')
    image_bytes = image_bytesio.getvalue()
    return b64encode(image_bytes)


if __name__ == "__main__":
    """
        Main execution block for the image transformation and prediction script.
        """
    # Set the API URL
    api_url = 'https://746f-209-137-198-8.ngrok-free.app/sdapi/v1/img2img'

    # ... (Provide inputs for radio, dict, img2img_prompt, and word_mask)

    # Call the predict function
    prediction_result = predict( image, img2img_prompt, word_mask, api_url)
    print("Prediction result:", prediction_result)
