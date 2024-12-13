import os
from typing import List

import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from starlette.responses import StreamingResponse

from api import file_uploader, invoke_model, convert_chat_completion_to_prompt, convert_response_to_openai, \
    to_json_string
from middleware import LimitRequestSizeMiddleware, lifespan
from models import PredictArgs, PredictResponse, ChatCompletionModel, ChatCompletionResponse

app = FastAPI(lifespan=lifespan)
app.add_middleware(LimitRequestSizeMiddleware)


def __return_response(request: PredictArgs,
                      stream=False,
                      is_openai=False,
                      background_tasks: BackgroundTasks = None):
    model, responses = invoke_model(request)
    if stream:
        background_tasks.add_task(model.unload_model_after_stream)
        if is_openai:
            streaming_responses = (to_json_string(convert_response_to_openai(
                predict_response=resp,
                task=request.task,
                is_stream=stream,
                model=request.model
            )) for resp in responses)
        else:
            streaming_responses = (to_json_string(resp).encode("utf-8") for resp in responses)
        return StreamingResponse(streaming_responses, media_type="application/json")
    else:
        if is_openai:
            return convert_response_to_openai(predict_response=responses,
                                              is_stream=False,
                                              task=request.task,
                                              model=request.model)
        else:
            return responses


@app.post("/v1/predict", response_model=List[PredictResponse])
async def predict(request: PredictArgs):
    return __return_response(request)


@app.post("/v1/predict_async")
async def predict_async(request: PredictArgs, background_tasks: BackgroundTasks):
    return __return_response(request, stream=True, background_tasks=background_tasks)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionModel, background_tasks: BackgroundTasks):
    predict_args = convert_chat_completion_to_prompt(request)
    return __return_response(request=predict_args, stream=request.stream, is_openai=True,
                             background_tasks=background_tasks)


@app.put("/v1/asset")
async def asset(files: List[UploadFile] = File(...)):
    return file_uploader().upload_batch(files)


if __name__ == "__main__":
    port = os.environ.get('PORT', '8000')
    uvicorn.run(app, host="0.0.0.0", port=int(port), log_config='conf/log.ini')
