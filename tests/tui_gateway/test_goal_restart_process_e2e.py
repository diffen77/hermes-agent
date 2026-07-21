"""Fresh-process restart recovery for durable ``/goal`` execution.

These tests intentionally drive ``tui_gateway.entry`` over its real line-delimited
stdio JSON-RPC transport and a real local OpenAI-compatible HTTP endpoint.  The
parent process opens state.db only to assert/mutate durable state between backend
processes; no gateway or supervisor implementation is imported into a child.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hermes_cli.goals import load_goal
from hermes_state import SessionDB


_REPO = Path(__file__).resolve().parents[2]
_GOAL = "Finish the restart recovery acceptance task"


class _FakeModelHandler(BaseHTTPRequestHandler):
    requests: queue.Queue[dict]
    initial_release: threading.Event
    work_count: int
    lock: threading.Lock

    def do_POST(self):  # noqa: N802 - http.server callback name
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        messages = body.get("messages") or []
        joined = "\n".join(
            str(message.get("content") or "")
            for message in messages
            if isinstance(message, dict)
        )
        is_judge = "strict judge evaluating whether an autonomous agent" in joined.lower()
        with type(self).lock:
            if not is_judge:
                type(self).work_count += 1
                work_number = type(self).work_count
            else:
                work_number = 0
        type(self).requests.put(
            {
                "body": body,
                "is_judge": is_judge,
                "work_number": work_number,
                "stream": body.get("stream"),
                "last_user": next(
                    (
                        str(message.get("content") or "")
                        for message in reversed(messages)
                        if isinstance(message, dict) and message.get("role") == "user"
                    ),
                    "",
                ),
            }
        )

        if is_judge:
            self._json_response('{"verdict":"done","reason":"acceptance achieved"}')
            return

        # The first actual agent request belongs to process A.  Hold the HTTP
        # handler after observation so SIGKILL is guaranteed to interrupt an
        # owned, durably-leased turn rather than a pre-request setup phase.
        if work_number == 1:
            type(self).initial_release.wait(timeout=20)
        self._stream_response("The acceptance task is completed and verified.")

    def _json_response(self, text: str) -> None:
        payload = {
            "id": "judge",
            "object": "chat.completion",
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _stream_response(self, text: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        chunks = [
            {
                "id": "work",
                "object": "chat.completion.chunk",
                "model": "fake-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "work",
                "object": "chat.completion.chunk",
                "model": "fake-model",
                "choices": [
                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                ],
            },
            {
                "id": "work",
                "object": "chat.completion.chunk",
                "model": "fake-model",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            },
        ]
        try:
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args) -> None:  # noqa: A002
        pass


@pytest.fixture
def fake_model():
    handler = type(
        "FakeModelHandler",
        (_FakeModelHandler,),
        {
            "requests": queue.Queue(),
            "initial_release": threading.Event(),
            "work_count": 0,
            "lock": threading.Lock(),
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, handler
    finally:
        handler.initial_release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class _GatewayProcess:
    def __init__(self, home: Path):
        env = os.environ.copy()
        env.update(
            {
                "HERMES_HOME": str(home),
                "HERMES_TEST_GOAL_LEASE_SECONDS": "2",
                "HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S": "3",
                "PYTHONPATH": str(_REPO) + os.pathsep + env.get("PYTHONPATH", ""),
            }
        )
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "tui_gateway.entry"],
            cwd=_REPO,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.stdout: queue.Queue[dict] = queue.Queue()
        self.stderr: list[str] = []
        self._next_id = 1
        threading.Thread(target=self._drain_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            try:
                self.stdout.put(json.loads(line))
            except json.JSONDecodeError:
                self.stderr.append(f"non-JSON stdout: {line!r}")

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        self.stderr.extend(self.proc.stderr)

    def wait_event(self, event_type: str, timeout: float = 8.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                frame = self.stdout.get(timeout=min(0.2, deadline - time.monotonic()))
            except queue.Empty:
                if self.proc.poll() is not None:
                    break
                continue
            if frame.get("method") == "event" and (
                frame.get("params") or {}
            ).get("type") == event_type:
                return frame
        raise AssertionError(
            f"timed out waiting for {event_type}; returncode={self.proc.poll()}\n"
            + "".join(self.stderr)
        )

    def rpc(self, method: str, params: dict, timeout: float = 8.0) -> dict:
        request_id = self._next_id
        self._next_id += 1
        assert self.proc.stdin is not None
        self.proc.stdin.write(
            json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            )
            + "\n"
        )
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                frame = self.stdout.get(timeout=min(0.2, deadline - time.monotonic()))
            except queue.Empty:
                if self.proc.poll() is not None:
                    break
                continue
            if frame.get("id") == request_id:
                assert "error" not in frame, frame
                return frame["result"]
        raise AssertionError(
            f"timed out waiting for RPC {method}; returncode={self.proc.poll()}\n"
            + "".join(self.stderr)
        )

    def crash(self) -> None:
        self.proc.kill()
        self.proc.wait(timeout=3)

    def close_cleanly(self) -> None:
        if self.proc.poll() is not None:
            return
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        self.proc.wait(timeout=8)
        assert self.proc.returncode == 0, "".join(self.stderr)

    def kill_if_alive(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=3)


def _write_config(home: Path, port: int) -> None:
    home.mkdir()
    endpoint = f"http://127.0.0.1:{port}/v1"
    (home / "config.yaml").write_text(
        "\n".join(
            [
                "model:",
                "  default: fake-model",
                "  provider: custom:restart-fake",
                "custom_providers:",
                "  - name: restart-fake",
                f"    base_url: {endpoint}",
                "    api_key: test-key",
                "    model: fake-model",
                "    api_mode: chat_completions",
                "auxiliary:",
                "  goal_judge:",
                "    provider: custom",
                "    model: fake-model",
                f"    base_url: {endpoint}",
                "    api_key: test-key",
                "goals:",
                "  max_turns: 4",
                "  recovery_max_concurrency: 1",
                "memory:",
                "  provider: none",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _next_request(handler, *, timeout: float = 8.0) -> dict:
    try:
        return handler.requests.get(timeout=timeout)
    except queue.Empty as exc:
        raise AssertionError("timed out waiting for fake-model request") from exc


def _wait_for_state(db: SessionDB, session_id: str, predicate, timeout: float = 8.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = load_goal(session_id, db=db)
        if last is not None and predicate(last):
            return last
        time.sleep(0.05)
    raise AssertionError(f"goal state did not converge; last={last}")


def _start_owned_goal(home: Path, handler) -> tuple[_GatewayProcess, str]:
    backend = _GatewayProcess(home)
    backend.wait_event("gateway.ready")
    created = backend.rpc(
        "session.create",
        {
            "source": "tui",
            "cwd": str(home),
            "model": "fake-model",
            "provider": "custom:restart-fake",
        },
    )
    sid = created["session_id"]
    stored_id = created["stored_session_id"]
    directive = backend.rpc(
        "command.dispatch", {"session_id": sid, "name": "/goal", "arg": _GOAL}
    )
    assert directive["type"] == "send"
    backend.rpc(
        "prompt.submit",
        {
            "session_id": sid,
            "text": directive["message"],
            "goal_activation": directive["goal_activation"],
        },
    )
    first = _next_request(handler)
    assert first["is_judge"] is False
    assert first["work_number"] == 1
    return backend, stored_id


@pytest.mark.parametrize("restart_mode", ["expired_lease", "timed_wait"])
def test_goal_recovers_in_fresh_gateway_without_client_input(
    tmp_path, fake_model, restart_mode
):
    server, handler = fake_model
    home = tmp_path / ".hermes"
    _write_config(home, server.server_address[1])
    backend_a = backend_b = None
    db = None
    try:
        backend_a, stored_id = _start_owned_goal(home, handler)
        db = SessionDB(home / "state.db")
        active = _wait_for_state(db, stored_id, lambda state: state.status == "active")
        lease = db.get_goal_execution_lease(stored_id)
        assert lease is not None
        assert lease["generation"] == active.generation
        assert lease["owner_id"]
        assert lease["claim_token"]

        backend_a.crash()
        handler.initial_release.set()

        if restart_mode == "timed_wait":
            from hermes_constants import reset_hermes_home_override, set_hermes_home_override
            from hermes_cli.goals import GoalManager

            token = set_hermes_home_override(home)
            try:
                waiting = GoalManager(stored_id).wait_for_seconds(
                    3, "restart deadline"
                )
            finally:
                reset_hermes_home_override(token)
            assert waiting.generation == active.generation + 1

        backend_b = _GatewayProcess(home)
        backend_b.wait_event("gateway.ready")
        # Deliberately no session.create, session.resume, prompt.submit, or any
        # other client RPC after ready: startup recovery owns all subsequent work.
        recovered = _next_request(handler, timeout=9.0)
        assert recovered["is_judge"] is False
        assert recovered["work_number"] == 2
        judge = _next_request(handler, timeout=5.0)
        assert judge["is_judge"] is True, (
            {key: judge[key] for key in ("work_number", "stream", "last_user")},
            "".join(backend_b.stderr)[-4000:],
        )

        done = _wait_for_state(
            db,
            stored_id,
            lambda state: state.status == "done",
            timeout=5.0,
        )
        assert done.last_verdict == "done"
        assert done.last_reason == "acceptance achieved"
        final_lease = db.get_goal_execution_lease(stored_id)
        assert final_lease is not None
        assert final_lease["owner_id"] is None
        assert final_lease["claim_token"] is None
        assert handler.work_count == 2  # initial crash + exactly one recovered turn

        transcript = db.get_messages_as_conversation(stored_id)
        roles = [message["role"] for message in transcript]
        assert roles == ["user", "assistant"], transcript
        assert transcript[0]["content"] == _GOAL
        assert "completed and verified" in transcript[1]["content"]

        backend_b.close_cleanly()
        backend_b = None
    finally:
        handler.initial_release.set()
        if db is not None:
            db.close()
        if backend_a is not None:
            backend_a.kill_if_alive()
        if backend_b is not None:
            backend_b.kill_if_alive()


def test_paused_new_generation_is_not_resurrected_after_crash(tmp_path, fake_model):
    server, handler = fake_model
    home = tmp_path / ".hermes"
    _write_config(home, server.server_address[1])
    backend_a = backend_b = None
    db = None
    try:
        backend_a, stored_id = _start_owned_goal(home, handler)
        db = SessionDB(home / "state.db")
        active = _wait_for_state(db, stored_id, lambda state: state.status == "active")
        backend_a.crash()
        handler.initial_release.set()

        # Canonical control transition: generation advances and atomically fences
        # the crashed backend's stale lease before process B exists.
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.goals import GoalManager

        token = set_hermes_home_override(home)
        try:
            paused = GoalManager(stored_id).pause("restart test pause")
        finally:
            reset_hermes_home_override(token)
        assert paused is not None
        assert paused.generation == active.generation + 1

        backend_b = _GatewayProcess(home)
        backend_b.wait_event("gateway.ready")
        with pytest.raises(queue.Empty):
            handler.requests.get(timeout=4.0)
        persisted = load_goal(stored_id, db=db)
        assert persisted is not None
        assert persisted.status == "paused"
        assert persisted.generation == paused.generation
        assert handler.work_count == 1
        backend_b.close_cleanly()
        backend_b = None
    finally:
        handler.initial_release.set()
        if db is not None:
            db.close()
        if backend_a is not None:
            backend_a.kill_if_alive()
        if backend_b is not None:
            backend_b.kill_if_alive()
