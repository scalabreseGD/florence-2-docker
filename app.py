import uuid
from typing import List

from fastapi import FastAPI
from datetime import datetime

from api import florence
from middleware import LimitRequestSizeMiddleware
from models import PredictArgs, PredictResponse

app = FastAPI()
app.add_middleware(LimitRequestSizeMiddleware, max_upload_size=100 * 1024 * 1024)


def __now_timestamp():
    curr_dt = datetime.now()
    return int(round(curr_dt.timestamp()))


def __return_response(request: PredictArgs) -> List[PredictResponse]:
    model = florence(request.model)
    responses = model.call_model(task=request.task, text=request.text, images=request.images)
    return [PredictResponse(task=request.task, response=resp[request.task]) for resp in responses]


@app.post("/v1/predict", response_model=List[PredictResponse])
async def completions(request: PredictArgs):
    return __return_response(request)
