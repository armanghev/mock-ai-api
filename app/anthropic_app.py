from __future__ import annotations

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from app.anthropic import builders, catalog
from app.scenarios import parse_scenario
from app.schemas import AnthropicMessageRequest

DEFAULT_PORT = 8012


async def health(_: Request) -> Response:
    return JSONResponse({"status": "ok", "provider": "anthropic"})


async def list_models(_: Request) -> Response:
    return JSONResponse(catalog.list_models())


async def retrieve_model(request: Request) -> Response:
    model = catalog.get_model(request.path_params["model_id"])
    if model is None:
        return JSONResponse(
            status_code=404,
            content={
                "type": "error",
                "error": {
                    "type": "not_found_error",
                    "message": f"Model '{request.path_params['model_id']}' not found",
                },
            },
        )
    return JSONResponse(model)


async def create_message(request: Request) -> Response:
    payload = AnthropicMessageRequest.model_validate(await request.json())
    scenario = parse_scenario(payload.model)
    error_response = catalog.maybe_raise_error(scenario)
    if error_response is not None:
        return JSONResponse(
            status_code=error_response["status_code"], content=error_response["body"]
        )

    if payload.stream:
        return StreamingResponse(
            builders.stream_message(scenario, payload.messages),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(builders.build_message(scenario, payload.messages))


async def count_message_tokens(request: Request) -> Response:
    payload = await request.json()
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    return JSONResponse(builders.count_tokens(messages))


async def create_batch(request: Request) -> Response:
    payload = await request.json()
    requests = payload.get("requests", [])
    if not isinstance(requests, list):
        requests = []
    batch = catalog.batch_store.create(requests)
    return JSONResponse(catalog.public_record(batch))


async def list_batches(_: Request) -> Response:
    batches = [catalog.public_record(batch) for batch in catalog.batch_store.list_all()]
    return JSONResponse(
        {
            "data": batches,
            "first_id": batches[0]["id"] if batches else None,
            "has_more": False,
            "last_id": batches[-1]["id"] if batches else None,
        }
    )


async def retrieve_batch(request: Request) -> Response:
    batch = catalog.batch_store.get(request.path_params["message_batch_id"])
    if batch is None:
        return JSONResponse(
            status_code=404,
            content={
                "type": "error",
                "error": {"type": "not_found_error", "message": "Batch not found"},
            },
        )
    return JSONResponse(catalog.public_record(batch))


async def cancel_batch(request: Request) -> Response:
    batch = catalog.batch_store.cancel(request.path_params["message_batch_id"])
    if batch is None:
        return JSONResponse(
            status_code=404,
            content={
                "type": "error",
                "error": {"type": "not_found_error", "message": "Batch not found"},
            },
        )
    return JSONResponse(catalog.public_record(batch))


async def delete_batch(request: Request) -> Response:
    deleted = catalog.batch_store.delete(request.path_params["message_batch_id"])
    if not deleted:
        return JSONResponse(
            status_code=404,
            content={
                "type": "error",
                "error": {"type": "not_found_error", "message": "Batch not found"},
            },
        )
    return Response(status_code=204)


async def batch_results(request: Request) -> Response:
    lines = catalog.batch_store.results(request.path_params["message_batch_id"])
    if lines is None:
        return JSONResponse(
            status_code=404,
            content={
                "type": "error",
                "error": {"type": "not_found_error", "message": "Batch not found"},
            },
        )

    async def jsonl_stream():
        import json

        for index, line in enumerate(lines):
            if index > 0:
                import asyncio

                await asyncio.sleep(0.04)
            yield json.dumps(line, separators=(",", ":")) + "\n"

    return StreamingResponse(jsonl_stream(), media_type="application/x-ndjson")


async def upload_file(request: Request) -> Response:
    form = await request.form()
    uploaded = form.get("file")
    filename = (
        getattr(uploaded, "filename", "document.pdf")
        if uploaded is not None
        else "document.pdf"
    )
    content_type = (
        getattr(uploaded, "content_type", "application/pdf")
        if uploaded is not None
        else "application/pdf"
    )
    raw = await uploaded.read() if uploaded is not None else b"%PDF-1.4 mock"
    record = catalog.file_store.create(filename, content_type, len(raw), raw)
    return JSONResponse(catalog.public_record(record))


async def list_files(_: Request) -> Response:
    files = [catalog.public_record(record) for record in catalog.file_store.list_all()]
    return JSONResponse(
        {
            "data": files,
            "first_id": files[0]["id"] if files else None,
            "has_more": False,
            "last_id": files[-1]["id"] if files else None,
        }
    )


async def retrieve_file(request: Request) -> Response:
    record = catalog.file_store.get(request.path_params["file_id"])
    if record is None:
        return JSONResponse(
            status_code=404,
            content={
                "type": "error",
                "error": {"type": "not_found_error", "message": "File not found"},
            },
        )
    return JSONResponse(catalog.public_record(record))


async def download_file(request: Request) -> Response:
    record = catalog.file_store.get(request.path_params["file_id"])
    if record is None:
        return JSONResponse(
            status_code=404,
            content={
                "type": "error",
                "error": {"type": "not_found_error", "message": "File not found"},
            },
        )
    return Response(content=record["_content"], media_type=record["mime_type"])


async def delete_file(request: Request) -> Response:
    deleted = catalog.file_store.delete(request.path_params["file_id"])
    if deleted is None:
        return JSONResponse(
            status_code=404,
            content={
                "type": "error",
                "error": {"type": "not_found_error", "message": "File not found"},
            },
        )
    return JSONResponse(deleted)


def create_app() -> Starlette:
    return Starlette(
        debug=True,
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/v1/models", list_models, methods=["GET"]),
            Route("/v1/models/{model_id}", retrieve_model, methods=["GET"]),
            Route("/v1/messages", create_message, methods=["POST"]),
            Route("/v1/messages/count_tokens", count_message_tokens, methods=["POST"]),
            Route("/v1/messages/batches", create_batch, methods=["POST"]),
            Route("/v1/messages/batches", list_batches, methods=["GET"]),
            Route(
                "/v1/messages/batches/{message_batch_id}",
                retrieve_batch,
                methods=["GET"],
            ),
            Route(
                "/v1/messages/batches/{message_batch_id}/cancel",
                cancel_batch,
                methods=["POST"],
            ),
            Route(
                "/v1/messages/batches/{message_batch_id}",
                delete_batch,
                methods=["DELETE"],
            ),
            Route(
                "/v1/messages/batches/{message_batch_id}/results",
                batch_results,
                methods=["GET"],
            ),
            Route("/v1/files", upload_file, methods=["POST"]),
            Route("/v1/files", list_files, methods=["GET"]),
            Route("/v1/files/{file_id}", retrieve_file, methods=["GET"]),
            Route("/v1/files/{file_id}/content", download_file, methods=["GET"]),
            Route("/v1/files/{file_id}", delete_file, methods=["DELETE"]),
        ],
    )


app = create_app()
