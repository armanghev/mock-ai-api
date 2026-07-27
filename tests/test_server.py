from __future__ import annotations

import sys

import server


class FakeProcess:
    instances: list["FakeProcess"] = []

    def __init__(self, *, target, args, name: str, daemon: bool = False) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.started = False
        self.terminated = False
        self.join_calls = 0
        FakeProcess.instances.append(self)

    def start(self) -> None:
        self.started = True

    def join(self) -> None:
        self.join_calls += 1
        if self.name == "openai-mock-api" and self.join_calls == 1:
            raise KeyboardInterrupt

    def terminate(self) -> None:
        self.terminated = True

    def is_alive(self) -> bool:
        return self.started and not self.terminated


def test_reload_children_are_non_daemonic_and_interrupt_terminates_them(
    monkeypatch,
) -> None:
    FakeProcess.instances = []
    monkeypatch.setattr(server.multiprocessing, "Process", FakeProcess)
    monkeypatch.setattr(sys, "argv", ["mock-ai-api", "--reload"])

    server.main()

    assert len(FakeProcess.instances) == 2
    assert all(not process.daemon for process in FakeProcess.instances)
    assert all(
        process.started and process.terminated for process in FakeProcess.instances
    )
