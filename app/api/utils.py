import base64
import io
import re
from typing import Tuple, Union

import numpy as np
from PIL import Image


def is_base64_string(string):
    # Check if the string is very long and only contains Base64 valid characters
    return (
            len(string) > 100  # Typical length for Base64 encoded data
            and re.fullmatch(r'[A-Za-z0-9+/=]+', string)  # Only Base64 characters
            and (len(string) % 4 == 0)  # Base64 strings are divisible by 4
    )


def base64_to_image_with_size(base64_string) -> (Image, Union[Tuple[int, int], np.ndarray]):
    image_data = base64.b64decode(base64_string)  # Decode the base64 string
    image = Image.open(io.BytesIO(image_data))  # Convert to PIL.Imag
    size = image.size
    return image, size


def load_image_from_path(path: str) -> Image:
    with open(path, 'rb') as f:
        image = Image.open(f)
        image.load()
    size = image.size
    return image, size
