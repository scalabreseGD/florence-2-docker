import base64
import io
import re
from typing import Tuple, Union, Optional, Callable, Dict, List, Any

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def is_base64_string(string):
    # Check if the string is very long and only contains Base64 valid characters
    return (
            len(string) > 100  # Typical length for Base64 encoded data
            and re.fullmatch(r'[A-Za-z0-9+/=]+', string)  # Only Base64 characters
            and (len(string) % 4 == 0)  # Base64 strings are divisible by 4
    )


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


def scale_image(image, scale_factor=None):
    if scale_factor <= 0:
        raise ValueError("Scale factor must be positive.")

    width_old, height_old = image.shape[1], image.shape[0]
    width_new = int(width_old * scale_factor)
    height_new = int(height_old * scale_factor)
    return cv2.resize(image, (width_new, height_new), interpolation=cv2.INTER_LINEAR)


def load_video_from_path(path: str,
                         scale_factor: Optional[float] = None,
                         start_second: Optional[int] = 0,
                         end_second: Optional[int] = None):
    def to_pil(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return image, image.size

    # Open the video file
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)

    # Check if the video opened successfully
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the start and end frames
    start_frame = int(start_second * fps)
    end_frame = int(end_second * fps) if end_second is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set the initial frame position to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    current_frame = start_frame

    # Read frames from start_frame to end_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()

        if not ret:
            print("Reached the end of the video or encountered an error.")
            break
        if scale_factor:
            frame = scale_image(frame, scale_factor)
        frame = to_pil(frame)
        # Append the frame to the list
        frames.append(frame)

        # Increment the frame count
        current_frame += 1

    # Release the video capture object
    cap.release()

    return frames


def perform_in_batch(images, batch_size, function: Callable[[List, Dict], Any], **kwargs):
    results = []
    for frame_index in tqdm(range(0, len(images), min(len(images), batch_size)), desc='Performing inference'):
        batch = function(images[frame_index:frame_index + batch_size], **kwargs)
        results.extend(batch)
    return results
