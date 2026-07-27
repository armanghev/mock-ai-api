from __future__ import annotations

import sys

import server


class FakeProcess:
    instances: list["FakeProcess"] = []

    def __init__(self, command: list[str], **kwargs) -> None:
        self.command = command
        self.kwargs = kwargs
        self.pid = 100 + len(FakeProcess.instances)
        self.terminated = False
        self.wait_calls = 0
        FakeProcess.instances.append(self)

    def wait(self) -> None:
        self.wait_calls += 1
        if "app.openai_app:app" in self.command and self.wait_calls == 1:
            raise KeyboardInterrupt

    def terminate(self) -> None:
        self.terminated = True

    def poll(self) -> None:
        return None if not self.terminated else 0


def test_reload_uses_top_level_uvicorn_processes_and_interrupt_terminates_them(
    monkeypatch,
) -> None:
    FakeProcess.instances = []
    monkeypatch.setattr(server.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(sys, "argv", ["mock-ai-api", "--reload"])
    terminated_groups: list[int] = []
    monkeypatch.setattr(
        server.os,
        "killpg",
        lambda pid, _signal: terminated_groups.append(pid),
    )

    server.main()

    assert len(FakeProcess.instances) == 2
    assert all(
        process.command[:3] == [sys.executable, "-m", "uvicorn"]
        for process in FakeProcess.instances
    )
    assert all("--reload" in process.command for process in FakeProcess.instances)
    assert all(
        process.kwargs == {"start_new_session": True}
        for process in FakeProcess.instances
    )
    assert terminated_groups == [100, 101]
