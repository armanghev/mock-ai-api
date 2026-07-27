from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys


from app.anthropic_app import DEFAULT_PORT as ANTHROPIC_PORT
from app.openai_app import DEFAULT_PORT as OPENAI_PORT


def server_command(app: str, port: int, reload: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        app,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "info",
    ]
    if reload:
        command.append("--reload")
    return command


def stop_process(process: subprocess.Popen[object]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass


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

    processes = [
        subprocess.Popen(
            server_command("app.openai_app:app", args.openai_port, args.reload),
            start_new_session=True,
        ),
        subprocess.Popen(
            server_command("app.anthropic_app:app", args.anthropic_port, args.reload),
            start_new_session=True,
        ),
    ]
    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        print("\nStopping mock API servers...")
    finally:
        for process in processes:
            stop_process(process)
        for process in processes:
            process.wait()


if __name__ == "__main__":
    main()
