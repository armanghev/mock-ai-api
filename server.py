from __future__ import annotations

import argparse
import multiprocessing

import uvicorn

from app.anthropic_app import DEFAULT_PORT as ANTHROPIC_PORT
from app.openai_app import DEFAULT_PORT as OPENAI_PORT


def run_openai_server(port: int, reload: bool) -> None:
    uvicorn.run(
        "app.openai_app:app",
        host="127.0.0.1",
        port=port,
        reload=reload,
        log_level="info",
    )


def run_anthropic_server(port: int, reload: bool) -> None:
    uvicorn.run(
        "app.anthropic_app:app",
        host="127.0.0.1",
        port=port,
        reload=reload,
        log_level="info",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenAI and Anthropic mock API servers"
    )
    parser.add_argument("--openai-port", type=int, default=OPENAI_PORT)
    parser.add_argument("--anthropic-port", type=int, default=ANTHROPIC_PORT)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    print(f"OpenAI mock API:    http://127.0.0.1:{args.openai_port}")
    print(f"Anthropic mock API: http://127.0.0.1:{args.anthropic_port}")

    openai_process = multiprocessing.Process(
        target=run_openai_server,
        args=(args.openai_port, args.reload),
        name="openai-mock-api",
    )
    anthropic_process = multiprocessing.Process(
        target=run_anthropic_server,
        args=(args.anthropic_port, args.reload),
        name="anthropic-mock-api",
    )

    openai_process.start()
    anthropic_process.start()
    try:
        openai_process.join()
        anthropic_process.join()
    except KeyboardInterrupt:
        print("\nStopping mock API servers...")
        for process in (openai_process, anthropic_process):
            if process.is_alive():
                process.terminate()
    finally:
        for process in (openai_process, anthropic_process):
            process.join()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
