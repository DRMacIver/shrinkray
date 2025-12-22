"""Tests for the subprocess protocol."""

import pytest

from shrinkray.subprocess.protocol import (
    ProgressUpdate,
    Request,
    Response,
    decode_bytes,
    deserialize,
    encode_bytes,
    serialize,
)


class TestRequest:
    def test_serialize_request(self):
        req = Request(id="123", command="start", params={"file": "test.txt"})
        result = serialize(req)
        assert '"id":"123"' in result
        assert '"command":"start"' in result
        assert '"params"' in result

    def test_deserialize_request(self):
        line = '{"id":"456","command":"status","params":{}}'
        result = deserialize(line)
        assert isinstance(result, Request)
        assert result.id == "456"
        assert result.command == "status"
        assert result.params == {}

    def test_request_roundtrip(self):
        original = Request(id="abc", command="cancel", params={"force": True})
        serialized = serialize(original)
        deserialized = deserialize(serialized)
        assert isinstance(deserialized, Request)
        assert deserialized.id == original.id
        assert deserialized.command == original.command
        assert deserialized.params == original.params


class TestResponse:
    def test_serialize_response_with_result(self):
        resp = Response(id="123", result={"status": "ok"})
        result = serialize(resp)
        assert '"id":"123"' in result
        assert '"result"' in result
        assert '"error":null' in result

    def test_serialize_response_with_error(self):
        resp = Response(id="123", error="Something went wrong")
        result = serialize(resp)
        assert '"id":"123"' in result
        assert '"error":"Something went wrong"' in result

    def test_deserialize_response_with_result(self):
        line = '{"id":"789","result":{"data":"value"},"error":null}'
        result = deserialize(line)
        assert isinstance(result, Response)
        assert result.id == "789"
        assert result.result == {"data": "value"}
        assert result.error is None

    def test_deserialize_response_with_error(self):
        line = '{"id":"789","error":"failed"}'
        result = deserialize(line)
        assert isinstance(result, Response)
        assert result.id == "789"
        assert result.error == "failed"

    def test_response_roundtrip(self):
        original = Response(id="xyz", result={"count": 42}, error=None)
        serialized = serialize(original)
        deserialized = deserialize(serialized)
        assert isinstance(deserialized, Response)
        assert deserialized.id == original.id
        assert deserialized.result == original.result
        assert deserialized.error == original.error


class TestProgressUpdate:
    def test_serialize_progress_update(self):
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

    def test_deserialize_progress_update(self):
        line = '{"type":"progress","data":{"status":"test","size":100,"original_size":200,"calls":5,"reductions":2}}'
        result = deserialize(line)
        assert isinstance(result, ProgressUpdate)
        assert result.status == "test"
        assert result.size == 100
        assert result.original_size == 200
        assert result.calls == 5
        assert result.reductions == 2

    def test_progress_update_roundtrip(self):
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


class TestBytesEncoding:
    def test_encode_bytes(self):
        data = b"Hello, World!"
        encoded = encode_bytes(data)
        assert isinstance(encoded, str)
        assert encoded == "SGVsbG8sIFdvcmxkIQ=="

    def test_decode_bytes(self):
        encoded = "SGVsbG8sIFdvcmxkIQ=="
        decoded = decode_bytes(encoded)
        assert decoded == b"Hello, World!"

    def test_bytes_roundtrip(self):
        original = b"\x00\x01\x02\xff\xfe\xfd"
        encoded = encode_bytes(original)
        decoded = decode_bytes(encoded)
        assert decoded == original

    def test_empty_bytes(self):
        original = b""
        encoded = encode_bytes(original)
        decoded = decode_bytes(encoded)
        assert decoded == original


class TestSerializeErrors:
    def test_serialize_invalid_type(self):
        with pytest.raises(TypeError):
            serialize("not a message")  # type: ignore

    def test_serialize_dict_raises(self):
        with pytest.raises(TypeError):
            serialize({"id": "123"})  # type: ignore
