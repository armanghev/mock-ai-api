from __future__ import annotations

from json import JSONDecodeError

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from app.common import extract_input_text
from app.errors import (
    InvalidRequestBody,
    openai_http_error,
    openai_validation_error,
    require_json_object,
)
from app.openai import builders, catalog
from app.scenarios import parse_scenario
from app.schemas import OpenAIChatCompletionRequest, OpenAILegacyCompletionRequest

DEFAULT_PORT = 8011


async def health(_: Request) -> Response:
    return JSONResponse({"status": "ok", "provider": "openai"})


async def list_models(_: Request) -> Response:
    return JSONResponse(catalog.list_models())


async def retrieve_model(request: Request) -> Response:
    model = catalog.get_model(request.path_params["model"])
    if model is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"Model '{request.path_params['model']}' not found",
                    "type": "invalid_request_error",
                }
            },
        )
    return JSONResponse(model)


async def create_chat_completion(request: Request) -> Response:
    payload = OpenAIChatCompletionRequest.model_validate(await request.json())
    scenario = parse_scenario(payload.model)
    error_response = catalog.maybe_raise_error(scenario)
    if error_response is not None:
        return JSONResponse(
            status_code=error_response["status_code"], content=error_response["body"]
        )

    if payload.stream:
        return StreamingResponse(
            builders.stream_chat_completion(scenario, payload.messages),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(builders.build_chat_completion(scenario, payload.messages))


async def create_legacy_completion(request: Request) -> Response:
    payload = OpenAILegacyCompletionRequest.model_validate(await request.json())
    scenario = parse_scenario(payload.model)
    error_response = catalog.maybe_raise_error(scenario)
    if error_response is not None:
        return JSONResponse(
            status_code=error_response["status_code"], content=error_response["body"]
        )

    prompt = (
        payload.prompt if isinstance(payload.prompt, str) else "\n".join(payload.prompt)
    )
    if payload.stream:
        return StreamingResponse(
            builders.stream_legacy_completion(scenario, prompt),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(builders.build_legacy_completion(scenario, prompt))


async def create_embeddings(request: Request) -> Response:
    payload = await require_json_object(request)
    return JSONResponse(
        builders.build_embeddings(
            payload.get("model", "text-embedding-3-small"), payload.get("input", "")
        )
    )


async def create_moderations(request: Request) -> Response:
    payload = await require_json_object(request)
    return JSONResponse(
        builders.build_moderation(
            payload.get("model", "omni-moderation-latest"),
            str(payload.get("input", "")),
        )
    )


async def create_image_generation(request: Request) -> Response:
    payload = await require_json_object(request)
    return JSONResponse(
        builders.build_image_generation(
            payload.get("model", "gpt-image-1"), str(payload.get("prompt", ""))
        )
    )


async def create_audio_speech(request: Request) -> Response:
    return Response(content=b"ID3mock-mp3-audio-bytes", media_type="audio/mpeg")


async def create_audio_transcription(request: Request) -> Response:
    return JSONResponse(
        {
            "text": "Hello, thanks for calling support.",
            "usage": {
                "type": "tokens",
                "input_tokens": 124,
                "output_tokens": 12,
                "total_tokens": 136,
            },
        }
    )


async def create_audio_translation(request: Request) -> Response:
    return JSONResponse({"text": "Hello, my name is Wolfgang and I come from Germany."})


async def create_response(request: Request) -> Response:
    payload = await require_json_object(request)
    scenario = parse_scenario(str(payload.get("model", "gpt-4.1")))
    error_response = catalog.maybe_raise_error(scenario)
    if error_response is not None:
        return JSONResponse(
            status_code=error_response["status_code"], content=error_response["body"]
        )

    input_value = payload.get("input", "")
    prompt_text = extract_input_text(input_value)
    output_text = builders.assistant_text(scenario, prompt_text)

    if payload.get("stream"):
        return StreamingResponse(
            builders.stream_response(scenario, input_value, output_text),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    record = catalog.build_response_record(
        model=str(payload.get("model", "gpt-4.1")),
        input_value=input_value,
        output_text=output_text,
        scenario=scenario,
    )
    public_record = {
        key: value for key, value in record.items() if not key.startswith("_")
    }
    catalog.response_store.save(record)
    return JSONResponse(public_record)


async def retrieve_response(request: Request) -> Response:
    response = catalog.response_store.get(request.path_params["response_id"])
    if response is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": "Response not found",
                    "type": "invalid_request_error",
                }
            },
        )
    public_record = {
        key: value for key, value in response.items() if not key.startswith("_")
    }
    return JSONResponse(public_record)


async def list_response_input_items(request: Request) -> Response:
    response = catalog.response_store.get(request.path_params["response_id"])
    if response is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": "Response not found",
                    "type": "invalid_request_error",
                }
            },
        )
    return JSONResponse(catalog.build_input_items(response))


async def cancel_response(request: Request) -> Response:
    response = catalog.response_store.cancel(request.path_params["response_id"])
    if response is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": "Response not found",
                    "type": "invalid_request_error",
                }
            },
        )
    public_record = {
        key: value for key, value in response.items() if not key.startswith("_")
    }
    return JSONResponse(public_record)


def create_app() -> Starlette:
    return Starlette(
        debug=True,
        exception_handlers={
            JSONDecodeError: openai_validation_error,
            InvalidRequestBody: openai_validation_error,
            ValidationError: openai_validation_error,
            HTTPException: openai_http_error,
        },
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/v1/models", list_models, methods=["GET"]),
            Route("/v1/models/{model}", retrieve_model, methods=["GET"]),
            Route("/v1/chat/completions", create_chat_completion, methods=["POST"]),
            Route("/v1/completions", create_legacy_completion, methods=["POST"]),
            Route("/v1/embeddings", create_embeddings, methods=["POST"]),
            Route("/v1/moderations", create_moderations, methods=["POST"]),
            Route("/v1/images/generations", create_image_generation, methods=["POST"]),
            Route("/v1/audio/speech", create_audio_speech, methods=["POST"]),
            Route(
                "/v1/audio/transcriptions", create_audio_transcription, methods=["POST"]
            ),
            Route("/v1/audio/translations", create_audio_translation, methods=["POST"]),
            Route("/v1/responses", create_response, methods=["POST"]),
            Route("/v1/responses/{response_id}", retrieve_response, methods=["GET"]),
            Route(
                "/v1/responses/{response_id}/input_items",
                list_response_input_items,
                methods=["GET"],
            ),
            Route(
                "/v1/responses/{response_id}/cancel", cancel_response, methods=["POST"]
            ),
        ],
    )


app = create_app()
