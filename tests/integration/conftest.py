from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import BinaryIO
from pathlib import Path

import httpx
import pytest


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RunningServer:
    process: subprocess.Popen[bytes]
    stderr: BinaryIO


def _reserve_local_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    sock.set_inheritable(True)
    return sock


def _start_server(app: str, sock: socket.socket) -> RunningServer:
    stderr = tempfile.TemporaryFile()
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                app,
                "--fd",
                str(sock.fileno()),
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            pass_fds=(sock.fileno(),),
            start_new_session=True,
        )
    except Exception:
        stderr.close()
        raise
    return RunningServer(process=process, stderr=stderr)


def _captured_stderr(server: RunningServer) -> str:
    server.stderr.seek(0)
    return server.stderr.read().decode(errors="replace")


def _wait_for_health(server: RunningServer, url: str) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            stderr = _captured_stderr(server)
            pytest.fail(f"Mock server exited before becoming healthy: {stderr}")
        try:
            if httpx.get(url, timeout=0.2).is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    pytest.fail(
        f"Mock server did not become healthy: {url}\n"
        f"Captured stderr:\n{_captured_stderr(server)}"
    )


def _stop_server(server: RunningServer) -> None:
    process = server.process
    if process.poll() is None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            process.wait(timeout=5)
    server.stderr.close()


@pytest.fixture(scope="module")
def mock_server_urls() -> Iterator[dict[str, str]]:
    if os.name != "posix":
        pytest.skip("The process-level SDK fixture requires POSIX FD inheritance")

    openai_socket = _reserve_local_socket()
    openai_port = int(openai_socket.getsockname()[1])
    try:
        openai_process = _start_server("app.openai_app:app", openai_socket)
    finally:
        openai_socket.close()

    anthropic_socket = _reserve_local_socket()
    anthropic_port = int(anthropic_socket.getsockname()[1])
    try:
        anthropic_process = _start_server("app.anthropic_app:app", anthropic_socket)
    except Exception:
        _stop_server(openai_process)
        raise
    finally:
        anthropic_socket.close()
    urls = {
        "openai": f"http://127.0.0.1:{openai_port}/v1",
        "anthropic": f"http://127.0.0.1:{anthropic_port}",
    }
    try:
        _wait_for_health(openai_process, f"http://127.0.0.1:{openai_port}/health")
        _wait_for_health(anthropic_process, f"http://127.0.0.1:{anthropic_port}/health")
        yield urls
    finally:
        _stop_server(openai_process)
        _stop_server(anthropic_process)


@pytest.fixture()
def openai_base_url(mock_server_urls: dict[str, str]) -> str:
    return mock_server_urls["openai"]


@pytest.fixture()
def anthropic_base_url(mock_server_urls: dict[str, str]) -> str:
    return mock_server_urls["anthropic"]
