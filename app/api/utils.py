import base64
import io
import re
from pathlib import Path

import cv2
from typing import Tuple, Union, Optional

import numpy as np
from PIL import Image


def is_base64_string(string):
    # Check if the string is very long and only contains Base64 valid characters
    return (
            len(string) > 100  # Typical length for Base64 encoded data
            and re.fullmatch(r'[A-Za-z0-9+/=]+', string)  # Only Base64 characters
            and (len(string) % 4 == 0)  # Base64 strings are divisible by 4
    )


def is_path_video(string: str) -> bool:
    file_path = Path("example/file.txt")
    return file_path.suffix[1:] == 'mp4'


def scale_image(image, scale_factor=None):
    if scale_factor is not None:
        # Resize the image
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        image = image.resize(new_size)
    return image


def base64_to_image_with_size(base64_string, scale_factor: Optional[float] = None) -> (
        Image, Union[Tuple[int, int], np.ndarray]):
    image_data = base64.b64decode(base64_string)  # Decode the base64 string
    image = Image.open(io.BytesIO(image_data))  # Convert to PIL.Imag
    scale_image(image, scale_factor)
    size = image.size
    return image, size


def load_image_from_path(path: str, scale_factor: Optional[float] = None) -> Image:
    with open(path, 'rb') as f:
        image = Image.open(f)
        scale_image(image, scale_factor)
        image.load()
    size = image.size
    return image, size


def load_video_from_path(path: str, scale_factor: Optional[float] = None):
    cap = cv2.VideoCapture(path)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert OpenCV image (BGR) to RGB and then to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        image = scale_image(pil_image, scale_factor)
        frames.append((image, image.size))
    return frames
