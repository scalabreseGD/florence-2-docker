import gc
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
from PIL.Image import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from api.patches import DEVICE, run_with_patch
from utils import base64_to_image_with_size, is_base64_string, load_image_from_path, \
    load_video_from_path, perform_in_batch

from models import PredictResponse


class Florence:

    def __init__(self,
                 model_name,
                 hf_token=None
                 ):
        self.hf_token = hf_token
        self.model_name = model_name
        self.model = None
        self.processor = None

    def call_model(self, task: str, text: str,
                   stream=False,
                   images: Optional[List[str]] = None,
                   video: Optional[str] = None,
                   batch_size: Optional[int] = None,
                   scale_factor=None,
                   start_second=None,
                   end_second=None):
        if self.processor is None:
            run_with_patch(self.__init_model)

        if text == '':
            text = task
        images_pillow_with_size = self.__read_images(images=images, video=video,
                                                     scale_factor=scale_factor,
                                                     start_second=start_second,
                                                     end_second=end_second)
        if stream:
            resp_generator = perform_in_batch(images_pillow_with_size, self.__call_model, True, task=task, text=text)
            only_res_generator = (resp[0] for resp in resp_generator)
            return ((PredictResponse(response=res).json() + "\n").encode("utf-8") for res in
                    only_res_generator)
        else:
            responses = perform_in_batch(images_pillow_with_size, self.__call_model, False, batch_size, task=task,
                                         text=text)
            return_resp = [PredictResponse(response=resp) for resp in responses]
            self.__unload_model()
            return return_resp

    def unload_model_after_stream(self):
        self.__unload_model()

    def __init_model(self):
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                          trust_remote_code=True,
                                                          torch_dtype=dtype,
                                                          token=self.hf_token
                                                          )
        self.model.to(DEVICE)
        self.processor = AutoProcessor.from_pretrained(self.model_name,
                                                       trust_remote_code=True,
                                                       token=self.hf_token,
                                                       clean_up_tokenization_spaces=True
                                                       )

    def __unload_model(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    @staticmethod
    def __read_images(images: Optional[List[str]] = None,
                      video: Optional[str] = None,
                      scale_factor=None,
                      start_second=None,
                      end_second=None):
        if video is not None:
            images_pillow_with_size = load_video_from_path(video, scale_factor, start_second, end_second)
        elif images is not None and is_base64_string(images[0]):
            images_pillow_with_size = [base64_to_image_with_size(image) for image in images]
        else:
            images_pillow_with_size = [load_image_from_path(image_path) for image_path in images]
        return images_pillow_with_size

    def __call_model(self, images: List[Tuple[Image, Union[Tuple[int, int], np.ndarray]]], task: str, text: str):
        inputs = self.processor(text=[text for _ in range(0, len(images))],
                                images=[images_pillow[0] for images_pillow in images],
                                return_tensors="pt").to(DEVICE)
        with torch.inference_mode(), torch.autocast(DEVICE.type):
            generated_ids = self.model.generate(
                input_ids=inputs.input_ids,
                pixel_values=inputs.pixel_values,
                max_new_tokens=1024,
                num_beams=5
            )
        gen_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
        responses = []
        for (image, image_size), generated_text in zip(images, gen_texts):
            response = self.processor.post_process_generation(generated_text, task=task,
                                                              image_size=image_size)
            responses.append(response)
            image.close()
        return responses


class FlorenceServe:
    loaded_models = {}

    def get_or_load_model(self, model_name, **kwargs):
        model = self.loaded_models.get(model_name)
        if not model:
            self.loaded_models[model_name] = Florence(model_name, **kwargs)
            return self.loaded_models[model_name]
        else:
            return model
