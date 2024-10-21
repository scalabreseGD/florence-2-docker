from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class LimitRequestSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        # Check if content length exceeds the max size
        if request.headers.get("content-length"):
            content_length = int(request.headers["content-length"])
            if content_length > self.max_upload_size:
                return Response(
                    "Request body is too large", status_code=413
                )

        response = await call_next(request)
        return response
