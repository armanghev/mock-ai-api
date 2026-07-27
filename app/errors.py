from __future__ import annotations

from typing import Any

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class InvalidRequestBody(Exception):
    """Raised when an endpoint requiring a JSON object receives another type."""


async def require_json_object(request: Request) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise InvalidRequestBody
    return payload


def openai_error(
    status_code: int, message: str, error_type: str = "invalid_request_error"
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type}},
    )


def anthropic_error(
    status_code: int, message: str, error_type: str = "invalid_request_error"
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": error_type, "message": message}},
    )


async def openai_validation_error(_: Request, __: Exception) -> Response:
    return openai_error(400, "Invalid request body")


async def anthropic_validation_error(_: Request, __: Exception) -> Response:
    return anthropic_error(400, "Invalid request body")


async def openai_http_error(_: Request, exception: Exception) -> Response:
    status_code = exception.status_code if isinstance(exception, HTTPException) else 500
    message = "Resource not found" if status_code == 404 else "Request failed"
    return openai_error(status_code, message)


async def anthropic_http_error(_: Request, exception: Exception) -> Response:
    status_code = exception.status_code if isinstance(exception, HTTPException) else 500
    message = "Resource not found" if status_code == 404 else "Request failed"
    error_type = "not_found_error" if status_code == 404 else "invalid_request_error"
    return anthropic_error(status_code, message, error_type)
