"""Tests for the subprocess protocol."""

import pytest

from shrinkray.subprocess.protocol import (
    PassStatsData,
    ProgressUpdate,
    Request,
    Response,
    decode_bytes,
    deserialize,
    encode_bytes,
    serialize,
)


# === Request tests ===


def test_serialize_request():
    req = Request(id="123", command="start", params={"file": "test.txt"})
    result = serialize(req)
    assert '"id":"123"' in result
    assert '"command":"start"' in result
    assert '"params"' in result


def test_deserialize_request():
    line = '{"id":"456","command":"status","params":{}}'
    result = deserialize(line)
    assert isinstance(result, Request)
    assert result.id == "456"
    assert result.command == "status"
    assert result.params == {}


def test_request_roundtrip():
    original = Request(id="abc", command="cancel", params={"force": True})
    serialized = serialize(original)
    deserialized = deserialize(serialized)
    assert isinstance(deserialized, Request)
    assert deserialized.id == original.id
    assert deserialized.command == original.command
    assert deserialized.params == original.params


# === Response tests ===


def test_serialize_response_with_result():
    resp = Response(id="123", result={"status": "ok"})
    result = serialize(resp)
    assert '"id":"123"' in result
    assert '"result"' in result
    assert '"error":null' in result


def test_serialize_response_with_error():
    resp = Response(id="123", error="Something went wrong")
    result = serialize(resp)
    assert '"id":"123"' in result
    assert '"error":"Something went wrong"' in result


def test_deserialize_response_with_result():
    line = '{"id":"789","result":{"data":"value"},"error":null}'
    result = deserialize(line)
    assert isinstance(result, Response)
    assert result.id == "789"
    assert result.result == {"data": "value"}
    assert result.error is None


def test_deserialize_response_with_error():
    line = '{"id":"789","error":"failed"}'
    result = deserialize(line)
    assert isinstance(result, Response)
    assert result.id == "789"
    assert result.error == "failed"


def test_response_roundtrip():
    original = Response(id="xyz", result={"count": 42}, error=None)
    serialized = serialize(original)
    deserialized = deserialize(serialized)
    assert isinstance(deserialized, Response)
    assert deserialized.id == original.id
    assert deserialized.result == original.result
    assert deserialized.error == original.error


# === ProgressUpdate tests ===


def test_serialize_progress_update():
    update = ProgressUpdate(
        status="Running pass X",
        size=1000,
        original_size=2000,
        calls=50,
        reductions=10,
    )
    result = serialize(update)
    assert '"type":"progress"' in result
    assert '"status":"Running pass X"' in result
    assert '"size":1000' in result
    assert '"original_size":2000' in result
    assert '"calls":50' in result
    assert '"reductions":10' in result


def test_deserialize_progress_update():
    line = '{"type":"progress","data":{"status":"test","size":100,"original_size":200,"calls":5,"reductions":2}}'
    result = deserialize(line)
    assert isinstance(result, ProgressUpdate)
    assert result.status == "test"
    assert result.size == 100
    assert result.original_size == 200
    assert result.calls == 5
    assert result.reductions == 2


def test_progress_update_roundtrip():
    original = ProgressUpdate(
        status="Working",
        size=500,
        original_size=1000,
        calls=25,
        reductions=5,
    )
    serialized = serialize(original)
    deserialized = deserialize(serialized)
    assert isinstance(deserialized, ProgressUpdate)
    assert deserialized.status == original.status
    assert deserialized.size == original.size
    assert deserialized.original_size == original.original_size
    assert deserialized.calls == original.calls
    assert deserialized.reductions == original.reductions


def test_progress_update_with_pass_stats():
    """Test serialization/deserialization of pass stats."""
    pass_stats = [
        PassStatsData(
            pass_name="test_pass",
            bytes_deleted=100,
            non_size_reductions=2,
            run_count=5,
            test_evaluations=50,
            successful_reductions=3,
            success_rate=60.0,
        )
    ]

    update = ProgressUpdate(
        status="Testing",
        size=100,
        original_size=200,
        calls=10,
        reductions=5,
        pass_stats=pass_stats,
    )

    serialized = serialize(update)
    deserialized = deserialize(serialized)

    assert isinstance(deserialized, ProgressUpdate)
    assert len(deserialized.pass_stats) == 1
    assert deserialized.pass_stats[0].pass_name == "test_pass"
    assert deserialized.pass_stats[0].bytes_deleted == 100
    assert deserialized.pass_stats[0].non_size_reductions == 2
    assert deserialized.pass_stats[0].run_count == 5
    assert deserialized.pass_stats[0].test_evaluations == 50
    assert deserialized.pass_stats[0].successful_reductions == 3
    assert deserialized.pass_stats[0].success_rate == 60.0


def test_progress_update_backward_compatibility():
    """Test that old messages without pass_stats still deserialize."""
    # Simulate old-style message without pass_stats field
    line = '{"type":"progress","data":{"status":"Running","size":100,"original_size":200,"calls":5,"reductions":2}}'

    deserialized = deserialize(line)

    assert isinstance(deserialized, ProgressUpdate)
    assert deserialized.pass_stats == []  # Empty list by default


def test_progress_update_with_multiple_pass_stats():
    """Test ProgressUpdate with multiple pass stats."""
    pass_stats = [
        PassStatsData(
            pass_name="hollow",
            bytes_deleted=500,
            non_size_reductions=0,
            run_count=3,
            test_evaluations=100,
            successful_reductions=2,
            success_rate=66.7,
        ),
        PassStatsData(
            pass_name="delete_duplicates",
            bytes_deleted=200,
            non_size_reductions=1,
            run_count=4,
            test_evaluations=80,
            successful_reductions=3,
            success_rate=75.0,
        ),
    ]

    update = ProgressUpdate(
        status="Running",
        size=300,
        original_size=1000,
        calls=20,
        reductions=8,
        pass_stats=pass_stats,
    )

    serialized = serialize(update)
    deserialized = deserialize(serialized)

    assert isinstance(deserialized, ProgressUpdate)
    assert len(deserialized.pass_stats) == 2
    assert deserialized.pass_stats[0].pass_name == "hollow"
    assert deserialized.pass_stats[0].bytes_deleted == 500
    assert deserialized.pass_stats[1].pass_name == "delete_duplicates"
    assert deserialized.pass_stats[1].bytes_deleted == 200


# === Bytes encoding tests ===


def test_encode_bytes():
    data = b"Hello, World!"
    encoded = encode_bytes(data)
    assert isinstance(encoded, str)
    assert encoded == "SGVsbG8sIFdvcmxkIQ=="


def test_decode_bytes():
    encoded = "SGVsbG8sIFdvcmxkIQ=="
    decoded = decode_bytes(encoded)
    assert decoded == b"Hello, World!"


def test_bytes_roundtrip():
    original = b"\x00\x01\x02\xff\xfe\xfd"
    encoded = encode_bytes(original)
    decoded = decode_bytes(encoded)
    assert decoded == original


def test_empty_bytes():
    original = b""
    encoded = encode_bytes(original)
    decoded = decode_bytes(encoded)
    assert decoded == original


# === Serialize error tests ===


def test_serialize_invalid_type():
    with pytest.raises(TypeError):
        serialize("not a message")  # type: ignore


def test_serialize_dict_raises():
    with pytest.raises(TypeError):
        serialize({"id": "123"})  # type: ignore
