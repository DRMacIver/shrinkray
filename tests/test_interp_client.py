"""Tests for WorkerClient, the socket-based client for the in-process worker.

Note: This file uses asyncio.run() because WorkerClient is built on asyncio
(for the textual TUI library), while the rest of the project uses trio.
Using asyncio.run() isolates these tests from the trio event loop.
"""

import asyncio
import socket
from contextlib import aclosing

import pytest

from shrinkray.interp.client import WorkerClient
from shrinkray.interp.protocol import (
    ProgressUpdate,
    Request,
    Response,
    deserialize,
    serialize,
)


def make_progress_update(
    *, size: int = 100, content_preview: str = ""
) -> ProgressUpdate:
    return ProgressUpdate(
        status="running",
        size=size,
        original_size=200,
        calls=5,
        reductions=2,
        content_preview=content_preview,
    )


class FakeWorkerEnd:
    """Drives the worker's end of the client's socketpair."""

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None

    async def start(self) -> None:
        self.reader, self.writer = await asyncio.open_connection(sock=self._sock)

    async def send(self, msg: Request | Response | ProgressUpdate) -> None:
        assert self.writer is not None
        self.writer.write((serialize(msg) + "\n").encode("utf-8"))
        await self.writer.drain()

    async def send_raw(self, data: bytes) -> None:
        assert self.writer is not None
        self.writer.write(data)
        await self.writer.drain()

    async def receive_request(self) -> Request:
        assert self.reader is not None
        line = await self.reader.readline()
        msg = deserialize(line.decode("utf-8"))
        assert isinstance(msg, Request)
        return msg

    async def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except (ConnectionError, BrokenPipeError):
                pass
        else:
            self._sock.close()


def make_client_and_worker() -> tuple[WorkerClient, FakeWorkerEnd]:
    client_sock, worker_sock = socket.socketpair()
    return WorkerClient(client_sock), FakeWorkerEnd(worker_sock)


async def start_both(client: WorkerClient, worker: FakeWorkerEnd) -> None:
    await client.start()
    await worker.start()


# === Construction and lifecycle ===


def test_worker_client_initial_state():
    client_sock, worker_sock = socket.socketpair()
    client = WorkerClient(client_sock)
    assert not client.is_completed
    assert client.error_message is None
    client_sock.close()
    worker_sock.close()


def test_worker_client_send_command_raises_without_start():
    async def run():
        client_sock, worker_sock = socket.socketpair()
        client = WorkerClient(client_sock)
        with pytest.raises(RuntimeError, match="not started"):
            await client.send_command("status")
        client_sock.close()
        worker_sock.close()

    asyncio.run(run())


def test_worker_client_close_before_start_closes_socket():
    async def run():
        client_sock, worker_sock = socket.socketpair()
        client = WorkerClient(client_sock)
        await client.close()
        # The worker end sees EOF immediately.
        worker_sock.settimeout(5)
        assert worker_sock.recv(1) == b""
        worker_sock.close()

    asyncio.run(run())


def test_worker_client_close_is_idempotent():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)
        await client.close()
        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_context_manager():
    async def run():
        client, worker = make_client_and_worker()
        async with client:
            await worker.start()
        await worker.close()

    asyncio.run(run())


def test_worker_client_close_signals_eof_to_worker():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)
        await client.close()
        assert worker.reader is not None
        line = await asyncio.wait_for(worker.reader.readline(), timeout=5)
        assert line == b""
        await worker.close()

    asyncio.run(run())


# === Command round trips ===


def test_worker_client_send_command_matches_response():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        async def respond():
            request = await worker.receive_request()
            assert request.command == "status"
            await worker.send(Response(id=request.id, result={"running": True}))

        responder = asyncio.create_task(respond())
        response = await asyncio.wait_for(client.send_command("status"), timeout=5)
        await responder
        assert response.result == {"running": True}

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_start_reduction_sends_expected_params():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        async def respond():
            request = await worker.receive_request()
            assert request.command == "start"
            assert request.params["file_path"] == "test.c"
            assert request.params["test"] == ["./test.sh"]
            assert request.params["parallelism"] == 2
            assert request.params["timeout"] == 5.0
            assert request.params["memory_limit"] == 1024
            await worker.send(Response(id=request.id, result={"status": "started"}))

        responder = asyncio.create_task(respond())
        response = await asyncio.wait_for(
            client.start_reduction(
                file_path="test.c",
                test=["./test.sh"],
                parallelism=2,
                timeout=5.0,
                memory_limit=1024,
            ),
            timeout=5,
        )
        await responder
        assert response.result == {"status": "started"}

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_start_reduction_omits_unset_optional_params():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        async def respond():
            request = await worker.receive_request()
            assert "parallelism" not in request.params
            assert "timeout" not in request.params
            assert "memory_limit" not in request.params
            await worker.send(Response(id=request.id, result={"status": "started"}))

        responder = asyncio.create_task(respond())
        await asyncio.wait_for(
            client.start_reduction(file_path="test.c", test=["./test.sh"]),
            timeout=5,
        )
        await responder

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_get_status():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        async def respond():
            request = await worker.receive_request()
            assert request.command == "status"
            await worker.send(Response(id=request.id, result={"running": False}))

        responder = asyncio.create_task(respond())
        response = await asyncio.wait_for(client.get_status(), timeout=5)
        await responder
        assert response.result == {"running": False}

        await client.close()
        await worker.close()

    asyncio.run(run())


# === Progress updates ===


def test_worker_client_yields_progress_updates():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        await worker.send(make_progress_update(size=150))
        await worker.send(make_progress_update(size=120))

        received = []
        async with aclosing(client.get_progress_updates()) as updates:
            async for update in updates:
                received.append(update)
                if len(received) == 2:
                    break

        assert [u.size for u in received] == [150, 120]
        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_progress_updates_stop_when_completed():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        await worker.send(Response(id="", result={"status": "completed"}))

        received = []
        async with aclosing(client.get_progress_updates()) as updates:
            async for update in updates:
                received.append(update)

        assert received == []
        assert client.is_completed
        await client.close()
        await worker.close()

    asyncio.run(run())


# === Completion and errors ===


def test_worker_client_completion_wakes_pending_futures():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        command = asyncio.create_task(client.send_command("status"))
        # Let the command be written before completing.
        await worker.receive_request()
        await worker.send(Response(id="", result={"status": "completed"}))

        with pytest.raises(Exception, match="completed"):
            await asyncio.wait_for(command, timeout=5)
        assert client.is_completed

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_error_response_sets_error_message():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        await worker.send(Response(id="", error="reduction failed"))

        deadline = asyncio.get_event_loop().time() + 5
        while not client.is_completed:
            assert asyncio.get_event_loop().time() < deadline
            await asyncio.sleep(0.01)
        assert client.error_message == "reduction failed"

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_error_response_wakes_pending_futures():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        command = asyncio.create_task(client.send_command("status"))
        await worker.receive_request()
        await worker.send(Response(id="", error="reduction failed"))

        with pytest.raises(Exception, match="reduction failed"):
            await asyncio.wait_for(command, timeout=5)

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_eof_before_completion_reports_error():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        await worker.close()

        deadline = asyncio.get_event_loop().time() + 5
        while not client.is_completed:
            assert asyncio.get_event_loop().time() < deadline
            await asyncio.sleep(0.01)
        assert client.error_message is not None
        assert "exited" in client.error_message

        await client.close()

    asyncio.run(run())


def test_worker_client_eof_wakes_pending_futures():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        command = asyncio.create_task(client.send_command("status"))
        await worker.receive_request()
        await worker.close()

        with pytest.raises(Exception, match="exited"):
            await asyncio.wait_for(command, timeout=5)

        await client.close()

    asyncio.run(run())


def test_worker_client_eof_after_completion_is_not_an_error():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        await worker.send(Response(id="", result={"status": "completed"}))
        await worker.close()

        deadline = asyncio.get_event_loop().time() + 5
        while not client.is_completed:
            assert asyncio.get_event_loop().time() < deadline
            await asyncio.sleep(0.01)
        # Give the reader task time to observe EOF as well.
        await asyncio.sleep(0.05)
        assert client.error_message is None

        await client.close()

    asyncio.run(run())


def test_worker_client_close_cancels_pending_futures():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        command = asyncio.create_task(client.send_command("status"))
        await worker.receive_request()
        await client.close()

        with pytest.raises(asyncio.CancelledError):
            await command

        await worker.close()

    asyncio.run(run())


# === Message parsing robustness ===


def test_worker_client_ignores_invalid_json():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        await worker.send_raw(b"not valid json\n")
        await worker.send(make_progress_update())

        async with aclosing(client.get_progress_updates()) as updates:
            async for update in updates:
                assert update.status == "running"
                break

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_ignores_request_messages():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        await worker.send(Request(id="1", command="status"))
        await worker.send(make_progress_update())

        async with aclosing(client.get_progress_updates()) as updates:
            async for update in updates:
                assert update.status == "running"
                break

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_ignores_unmatched_response():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        await worker.send(Response(id="no-such-request", result={}))
        await worker.send(make_progress_update())

        async with aclosing(client.get_progress_updates()) as updates:
            async for update in updates:
                assert update.status == "running"
                break

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_handles_messages_split_across_chunks():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        line = (serialize(make_progress_update()) + "\n").encode("utf-8")
        await worker.send_raw(line[:10])
        await asyncio.sleep(0.05)
        await worker.send_raw(line[10:])

        async with aclosing(client.get_progress_updates()) as updates:
            async for update in updates:
                assert update.size == 100
                break

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_handles_large_messages():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        # Larger than any fixed read buffer: the default asyncio stream
        # limit is 64KB and content previews alone can reach 100KB.
        await worker.send(make_progress_update(content_preview="x" * 500_000))

        async with aclosing(client.get_progress_updates()) as updates:
            async for update in updates:
                assert len(update.content_preview) == 500_000
                break

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_skips_blank_lines():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        await worker.send_raw(b"\n\n")
        await worker.send(make_progress_update())

        async with aclosing(client.get_progress_updates()) as updates:
            async for update in updates:
                assert update.status == "running"
                break

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_read_errors_are_treated_as_worker_exit():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        assert client._reader is not None

        async def explode(n: int = -1) -> bytes:
            raise OSError("read failed")

        client._reader.read = explode
        # Trip the reader loop by sending something.
        await worker.send(make_progress_update())

        deadline = asyncio.get_event_loop().time() + 5
        while not client.is_completed:
            assert asyncio.get_event_loop().time() < deadline
            await asyncio.sleep(0.01)
        assert client.error_message is not None
        assert "exited" in client.error_message

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_completion_skips_already_done_futures():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        done_future: asyncio.Future = asyncio.get_event_loop().create_future()
        done_future.set_result(Response(id="done", result={}))
        client._pending_responses["done"] = done_future

        await worker.send(Response(id="", result={"status": "completed"}))
        while not client.is_completed:
            await asyncio.sleep(0.01)
        assert done_future.result().id == "done"

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_error_skips_already_done_futures():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        done_future: asyncio.Future = asyncio.get_event_loop().create_future()
        done_future.set_result(Response(id="done", result={}))
        client._pending_responses["done"] = done_future

        await worker.send(Response(id="", error="boom"))
        while not client.is_completed:
            await asyncio.sleep(0.01)
        assert done_future.result().id == "done"

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_eof_skips_already_done_futures():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        done_future: asyncio.Future = asyncio.get_event_loop().create_future()
        done_future.set_result(Response(id="done", result={}))
        client._pending_responses["done"] = done_future

        await worker.close()
        while not client.is_completed:
            await asyncio.sleep(0.01)
        assert done_future.result().id == "done"

        await client.close()

    asyncio.run(run())


def test_worker_client_response_for_already_done_future_is_ignored():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        done_future: asyncio.Future = asyncio.get_event_loop().create_future()
        done_future.set_result(Response(id="done", result={"original": True}))
        client._pending_responses["done"] = done_future

        await worker.send(Response(id="done", result={"replacement": True}))
        await worker.send(make_progress_update())
        async with aclosing(client.get_progress_updates()) as updates:
            async for update in updates:
                assert update.status == "running"
                break
        assert done_future.result().result == {"original": True}

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_close_skips_done_futures():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        done_future: asyncio.Future = asyncio.get_event_loop().create_future()
        done_future.set_result(Response(id="done", result={}))
        client._pending_responses["done"] = done_future

        await client.close()
        assert done_future.result().id == "done"

        await worker.close()

    asyncio.run(run())


def test_worker_client_ignores_unsolicited_response_with_no_error_or_completion():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        await worker.send(Response(id="", result={"status": "something_else"}))
        await worker.send(make_progress_update())
        async with aclosing(client.get_progress_updates()) as updates:
            async for update in updates:
                assert update.status == "running"
                break
        assert not client.is_completed
        assert client.error_message is None

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_close_tolerates_wait_closed_errors():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        assert client._writer is not None

        async def explode():
            raise OSError("close failed")

        client._writer.wait_closed = explode
        await client.close()

        await worker.close()

    asyncio.run(run())


# === Convenience command wrappers ===


def test_worker_client_cancel_when_completed():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)
        await worker.send(Response(id="", result={"status": "completed"}))
        while not client.is_completed:
            await asyncio.sleep(0.01)
        response = await client.cancel()
        assert response.result["status"] == "already_completed"
        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_cancel_when_not_started():
    async def run():
        client, worker = make_client_and_worker()
        response = await client.cancel()
        assert response.result["status"] == "not_running"
        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_cancel_when_closed():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)
        await client.close()
        response = await client.cancel()
        assert response.result["status"] == "not_running"
        await worker.close()

    asyncio.run(run())


def test_worker_client_cancel_sends_command():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        async def respond():
            request = await worker.receive_request()
            assert request.command == "cancel"
            await worker.send(Response(id=request.id, result={"status": "cancelled"}))

        responder = asyncio.create_task(respond())
        response = await asyncio.wait_for(client.cancel(), timeout=5)
        await responder
        assert response.result["status"] == "cancelled"

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_cancel_swallows_send_errors():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        command = asyncio.create_task(client.cancel())
        await worker.receive_request()
        await worker.close()
        response = await asyncio.wait_for(command, timeout=5)
        assert response.result["status"] == "cancelled"

        await client.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "method,command",
    [
        pytest.param("disable_pass", "disable_pass", id="disable"),
        pytest.param("enable_pass", "enable_pass", id="enable"),
    ],
)
def test_worker_client_pass_control_sends_command(method, command):
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        async def respond():
            request = await worker.receive_request()
            assert request.command == command
            assert request.params == {"pass_name": "some_pass"}
            await worker.send(Response(id=request.id, result={"status": "ok"}))

        responder = asyncio.create_task(respond())
        response = await asyncio.wait_for(
            getattr(client, method)("some_pass"), timeout=5
        )
        await responder
        assert response.result["status"] == "ok"

        await client.close()
        await worker.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "method,args",
    [
        pytest.param("disable_pass", ("some_pass",), id="disable"),
        pytest.param("enable_pass", ("some_pass",), id="enable"),
        pytest.param("skip_current_pass", (), id="skip"),
    ],
)
def test_worker_client_pass_control_when_completed(method, args):
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)
        await worker.send(Response(id="", result={"status": "completed"}))
        while not client.is_completed:
            await asyncio.sleep(0.01)
        response = await getattr(client, method)(*args)
        assert response.result["status"] == "already_completed"
        await client.close()
        await worker.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "method,args,error",
    [
        pytest.param("disable_pass", ("some_pass",), "disable", id="disable"),
        pytest.param("enable_pass", ("some_pass",), "enable", id="enable"),
        pytest.param("skip_current_pass", (), "skip", id="skip"),
    ],
)
def test_worker_client_pass_control_send_errors_become_error_responses(
    method, args, error
):
    async def run():
        client, worker = make_client_and_worker()
        response = await getattr(client, method)(*args)
        assert response.error is not None
        assert error.lower() in response.error.lower()
        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_skip_current_pass_sends_command():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        async def respond():
            request = await worker.receive_request()
            assert request.command == "skip_pass"
            await worker.send(Response(id=request.id, result={"status": "skipped"}))

        responder = asyncio.create_task(respond())
        response = await asyncio.wait_for(client.skip_current_pass(), timeout=5)
        await responder
        assert response.result["status"] == "skipped"

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_restart_from_sends_command():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)

        async def respond():
            request = await worker.receive_request()
            assert request.command == "restart_from"
            assert request.params == {"reduction_number": 3}
            await worker.send(Response(id=request.id, result={"status": "restarted"}))

        responder = asyncio.create_task(respond())
        response = await asyncio.wait_for(client.restart_from(3), timeout=5)
        await responder
        assert response.result["status"] == "restarted"

        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_restart_from_when_completed():
    async def run():
        client, worker = make_client_and_worker()
        await start_both(client, worker)
        await worker.send(Response(id="", result={"status": "completed"}))
        while not client.is_completed:
            await asyncio.sleep(0.01)
        response = await client.restart_from(3)
        assert response.error is not None
        await client.close()
        await worker.close()

    asyncio.run(run())


def test_worker_client_restart_from_send_errors_become_error_responses():
    async def run():
        client, worker = make_client_and_worker()
        response = await client.restart_from(3)
        assert response.error is not None
        await client.close()
        await worker.close()

    asyncio.run(run())
