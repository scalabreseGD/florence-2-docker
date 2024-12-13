import json
import time
import uuid
from functools import lru_cache
from typing import Union, List

from starlette.responses import Response
import toml

from models import PredictArgs, ChatCompletionModel, PredictResponse, ChatCompletionResponse
from models.openai import ChatCompletionChoice, ChatCompletionMessage
from utils import parse_tags_from_content, file_type
from .florence import FlorenceServe, Florence
from .storage import FileUploader

__florence_serve = None
__file_uploader = None


def florence(model, **kwargs) -> Florence:
    global __florence_serve
    if __florence_serve is None:
        __florence_serve = FlorenceServe()
    return __florence_serve.get_or_load_model(model, **kwargs)


def file_uploader() -> FileUploader:
    global __file_uploader
    if __file_uploader is None:
        conf = read_conf()
        __file_uploader = FileUploader(base_path=conf['file_uploader']['base_path'])
    return __file_uploader


@lru_cache(maxsize=1)
def read_conf():
    with open("conf/conf.toml", "r") as f:
        data = toml.load(f)
    return data


def invoke_model(request: PredictArgs, stream=False):
    if request.video is not None and request.images is not None:
        Response(
            "Cannot use both images and video in the same request", status_code=400
        )
    model = florence(request.model)
    responses = model.call_model(task=request.task,
                                 text=request.text,
                                 images=request.images,
                                 stream=stream,
                                 video=request.video,
                                 batch_size=request.batch_size,
                                 scale_factor=request.scale_factor,
                                 start_second=request.start_second,
                                 end_second=request.end_second)
    return model, responses


def convert_chat_completion_to_prompt(chat_completion: ChatCompletionModel) -> PredictArgs:
    content = chat_completion.messages[0].content
    task, text, media_path = [value[1] for value in parse_tags_from_content(content)]
    media_type = file_type(media_path)
    if media_type == 'video':
        is_video = True
    elif media_type == 'image':
        is_video = False
    else:
        raise ValueError(f"{media_type} not supported")
    return PredictArgs(model=chat_completion.model,
                       task=task,
                       text=text if text else '',
                       video=media_path if is_video else None,
                       images=[media_path] if not is_video else [])


def __parse_content(predict_response, task):
    if isinstance(predict_response, PredictResponse):
        task_response = predict_response.response[task]
    else:
        task_response = predict_response[0].response[task]

    if isinstance(task_response, dict):
        return json.dumps(task_response)
    else:
        return task_response


def convert_response_to_openai(predict_response: Union[PredictResponse, List[PredictResponse]], is_stream,
                               task,
                               model) -> ChatCompletionResponse:
    if is_stream:
        _object = "chat.completion.chunk"
        _choices = ChatCompletionChoice(
            index=0,
            delta=ChatCompletionMessage(
                role='assistant',
                content=__parse_content(predict_response, task)),
            message=None
        )
    else:
        _object = "chat.completion"
        _choices = ChatCompletionChoice(
            index=0,
            message=ChatCompletionMessage(
                role='assistant',
                content=__parse_content(predict_response, task)
            ),
            delta=None
        )
    return ChatCompletionResponse(id=str(uuid.uuid4()),
                                  object=_object,
                                  created=round(time.time() * 1000),
                                  model=model,
                                  choices=[_choices]
                                  )


def to_json_string(model):
    return model.model_dump_json() + "\n"
