import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def base64_to_image_with_size(base64_string) -> (Image, np.ndarray):
    # Ensure no extra whitespace or newlines in base64 string
    base64_string = base64_string.strip()

    # Decode the base64 string into bytes
    decoded_bytes = base64.b64decode(base64_string)

    # Convert the decoded bytes to a NumPy array (flat array)
    buf = np.asarray(bytearray(decoded_bytes), dtype=np.uint8)

    # Try to decode the image using OpenCV
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Check if the image was properly decoded
    image_decoded_successfully = image is not None

    if image_decoded_successfully:
        # Get the image size (height, width)
        image_size = image.shape[:2]

        # Convert the image from BGR (OpenCV) to RGB (Pillow)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to Pillow Image
        pil_image = Image.fromarray(image_rgb)

        pil_image.show()  # Showing the image to check if it was decoded correctly
    else:
        image_size = None

    return image_decoded_successfully, image_size
