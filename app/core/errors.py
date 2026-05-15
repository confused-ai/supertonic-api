"""Consistent JSON error responses for all exception types."""

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import logger


def _error_body(
    message: str,
    error_type: str = "server_error",
    param: str | None = None,
    code: str | None = None,
    request_id: str | None = None,
) -> dict:
    """OpenAI-compatible error envelope.

    Shape: {"error": {"message": "...", "type": "...", "param": null, "code": null}}
    """
    err: dict = {
        "message": message,
        "type": error_type,
        "param": param,
        "code": code,
    }
    if request_id:
        err["request_id"] = request_id
    return {"error": err}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        error_type = "invalid_request_error" if exc.status_code < 500 else "server_error"
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(str(exc.detail), error_type=error_type, request_id=rid),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        errors = exc.errors()
        first = errors[0] if errors else {}
        loc = first.get("loc", ())
        # param = last non-"body" segment of the location path (matches OpenAI convention)
        param = str(loc[-1]) if loc and loc[-1] != "body" else None
        msg = first.get("msg", "validation error")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_body(
                msg,
                error_type="invalid_request_error",
                param=param,
                code="invalid_value",
                request_id=rid,
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body("Internal server error", request_id=rid),
        )
