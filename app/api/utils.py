import base64
import io
from typing import Tuple, Union

import numpy as np
from PIL import Image


def base64_to_image_with_size(base64_string) -> (Image, Union[Tuple[int, int], np.ndarray]):
    image_data = base64.b64decode(base64_string)  # Decode the base64 string
    image = Image.open(io.BytesIO(image_data))  # Convert to PIL.Imag
    size = image.size
    return image, size
