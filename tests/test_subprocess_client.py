"""Tests for SubprocessClient.

Note: This file uses asyncio.run() because SubprocessClient is built on asyncio
(for the textual TUI library), while the rest of the project uses trio.
Using asyncio.run() isolates these tests from the trio event loop.
"""

import asyncio

import pytest

from shrinkray.subprocess.client import SubprocessClient
from shrinkray.subprocess.protocol import ProgressUpdate, Response


class TestSubprocessClientUnit:
    """Unit tests for SubprocessClient (without actual subprocess)."""

    def test_initial_state(self):
        client = SubprocessClient()
        assert client._process is None
        assert client._completed is False
        assert client.is_completed is False

    def test_cancel_returns_early_when_completed(self):
        async def run():
            client = SubprocessClient()
            client._completed = True
            response = await client.cancel()
            assert response.result["status"] == "already_completed"

        asyncio.run(run())

    def test_cancel_returns_early_when_process_exited(self):
        async def run():
            client = SubprocessClient()

            # Mock a process that has already exited
            class MockProcess:
                returncode = 0

            client._process = MockProcess()  # type: ignore
            response = await client.cancel()
            assert response.result["status"] == "process_exited"

        asyncio.run(run())

    def test_send_command_raises_without_start(self):
        async def run():
            client = SubprocessClient()
            with pytest.raises(RuntimeError, match="not started"):
                await client.send_command("status")

        asyncio.run(run())

    def test_handle_message_ignores_invalid_json(self):
        async def run():
            client = SubprocessClient()
            # Should not raise
            await client._handle_message("not valid json")

        asyncio.run(run())

    def test_handle_message_queues_progress_updates(self):
        async def run():
            client = SubprocessClient()
            # Create a valid progress update JSON (data is nested under "data" key)
            progress_json = '{"type": "progress", "data": {"status": "running", "size": 100, "original_size": 200, "calls": 5, "reductions": 2, "interesting_calls": 3, "wasted_calls": 0, "runtime": 1.5, "parallel_workers": 1, "average_parallelism": 1.0, "effective_parallelism": 1.0, "time_since_last_reduction": 0.5, "content_preview": "test", "hex_mode": false}}'
            await client._handle_message(progress_json)

            # Check that progress was queued
            update = await asyncio.wait_for(client._progress_queue.get(), timeout=1.0)
            assert isinstance(update, ProgressUpdate)
            assert update.status == "running"
            assert update.size == 100

        asyncio.run(run())

    def test_handle_message_handles_completion(self):
        async def run():
            client = SubprocessClient()

            # Create a pending future to test wake-up
            loop = asyncio.get_event_loop()
            future: asyncio.Future = loop.create_future()
            client._pending_responses["test-id"] = future

            # Send completion signal
            completion_json = '{"id": "", "result": {"status": "completed"}, "error": null}'
            await client._handle_message(completion_json)

            assert client._completed is True
            assert client.is_completed is True

            # Future should have been woken up with an exception
            with pytest.raises(Exception, match="completed"):
                await asyncio.wait_for(future, timeout=1.0)

        asyncio.run(run())

    def test_handle_message_matches_response_to_pending(self):
        async def run():
            client = SubprocessClient()

            # Create a pending future
            loop = asyncio.get_event_loop()
            future: asyncio.Future = loop.create_future()
            client._pending_responses["test-id"] = future

            # Send matching response
            response_json = '{"id": "test-id", "result": {"key": "value"}, "error": null}'
            await client._handle_message(response_json)

            # Future should be resolved
            result = await asyncio.wait_for(future, timeout=1.0)
            assert isinstance(result, Response)
            assert result.result == {"key": "value"}

            # Pending response should be removed
            assert "test-id" not in client._pending_responses

        asyncio.run(run())


class TestSubprocessClientIntegration:
    """Integration tests that launch actual subprocesses."""

    def test_start_and_close(self):
        async def run():
            client = SubprocessClient()
            await client.start()
            assert client._process is not None
            await client.close()

        asyncio.run(run())

    def test_context_manager(self):
        async def run():
            async with SubprocessClient() as client:
                assert client._process is not None
            # After exiting context, process should be cleaned up

        asyncio.run(run())

    def test_get_status_before_start(self):
        async def run():
            async with SubprocessClient() as client:
                response = await client.get_status()
                assert response.result is not None
                assert response.result.get("running") is False

        asyncio.run(run())

    def test_close_handles_already_closed(self):
        async def run():
            client = SubprocessClient()
            # Should not raise when closing without starting
            await client.close()

        asyncio.run(run())

    def test_close_terminates_process(self):
        async def run():
            client = SubprocessClient()
            await client.start()
            assert client._process is not None
            await client.close()
            # Process should have been terminated

        asyncio.run(run())

    def test_get_progress_updates_stops_when_completed(self):
        async def run():
            client = SubprocessClient()
            client._completed = True

            updates = []
            async for update in client.get_progress_updates():
                updates.append(update)

            # Should exit immediately since _completed is True
            assert updates == []

        asyncio.run(run())


class TestSubprocessClientEdgeCases:
    """Edge case tests for SubprocessClient."""

    def test_cancel_after_process_exit(self):
        async def run():
            client = SubprocessClient()
            await client.start()

            # Simulate process exit
            if client._process is not None:
                client._process.terminate()
                await asyncio.wait_for(client._process.wait(), timeout=5.0)

            # Cancel should handle this gracefully
            response = await client.cancel()
            assert response.result is not None

            await client.close()

        asyncio.run(run())

    def test_multiple_close_calls(self):
        async def run():
            client = SubprocessClient()
            await client.start()
            await client.close()
            # Second close should not raise
            await client.close()

        asyncio.run(run())
