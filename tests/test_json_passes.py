"""Unit tests for JSON passes and utilities."""

import json

import pytest

from shrinkray.passes.definitions import ParseError
from shrinkray.passes.json import (
    JSON,
    DeleteIdentifiers,
    delete_identifiers,
    gather_identifiers,
    is_json,
)
from shrinkray.problem import BasicReductionProblem
from shrinkray.work import WorkContext


# =============================================================================
# is_json function tests
# =============================================================================


def test_is_json_valid_object():
    assert is_json(b'{"key": "value"}')


def test_is_json_valid_array():
    assert is_json(b'[1, 2, 3]')


def test_is_json_valid_string():
    assert is_json(b'"hello"')


def test_is_json_valid_number():
    assert is_json(b'42')


def test_is_json_valid_boolean():
    assert is_json(b'true')
    assert is_json(b'false')


def test_is_json_valid_null():
    assert is_json(b'null')


def test_is_json_invalid():
    assert not is_json(b'not json')


def test_is_json_invalid_incomplete():
    assert not is_json(b'{"key":')


def test_is_json_empty():
    assert not is_json(b'')


# =============================================================================
# JSON Format class tests
# =============================================================================


def test_json_format_repr():
    assert repr(JSON) == "JSON"


def test_json_format_name():
    assert JSON.name == "JSON"


def test_json_format_parse_object():
    result = JSON.parse(b'{"key": "value"}')
    assert result == {"key": "value"}


def test_json_format_parse_array():
    result = JSON.parse(b'[1, 2, 3]')
    assert result == [1, 2, 3]


def test_json_format_parse_nested():
    result = JSON.parse(b'{"a": [1, {"b": 2}]}')
    assert result == {"a": [1, {"b": 2}]}


def test_json_format_parse_invalid():
    with pytest.raises(ParseError):
        JSON.parse(b'not json')


def test_json_format_parse_invalid_unicode():
    with pytest.raises(ParseError):
        JSON.parse(b'\xff\xfe')


def test_json_format_dumps_object():
    result = JSON.dumps({"key": "value"})
    assert json.loads(result) == {"key": "value"}


def test_json_format_dumps_array():
    result = JSON.dumps([1, 2, 3])
    assert json.loads(result) == [1, 2, 3]


def test_json_format_roundtrip():
    original = {"nested": [1, 2, {"key": "value"}]}
    dumped = JSON.dumps(original)
    parsed = JSON.parse(dumped)
    assert parsed == original


# =============================================================================
# gather_identifiers function tests
# =============================================================================


def test_gather_identifiers_simple_dict():
    result = gather_identifiers({"a": 1, "b": 2})
    assert result == {"a", "b"}


def test_gather_identifiers_nested_dict():
    result = gather_identifiers({"a": {"b": {"c": 1}}})
    assert result == {"a", "b", "c"}


def test_gather_identifiers_list():
    result = gather_identifiers([1, 2, 3])
    assert result == set()


def test_gather_identifiers_list_with_dicts():
    result = gather_identifiers([{"a": 1}, {"b": 2}])
    assert result == {"a", "b"}


def test_gather_identifiers_mixed():
    result = gather_identifiers({"outer": [{"inner": 1}, {"nested": {"deep": 2}}]})
    assert result == {"outer", "inner", "nested", "deep"}


def test_gather_identifiers_empty_dict():
    result = gather_identifiers({})
    assert result == set()


def test_gather_identifiers_empty_list():
    result = gather_identifiers([])
    assert result == set()


def test_gather_identifiers_scalar():
    result = gather_identifiers(42)
    assert result == set()


def test_gather_identifiers_string():
    result = gather_identifiers("hello")
    assert result == set()


# =============================================================================
# DeleteIdentifiers Patches class tests
# =============================================================================


def test_delete_identifiers_patches_empty():
    di = DeleteIdentifiers()
    assert di.empty == frozenset()


def test_delete_identifiers_patches_combine():
    di = DeleteIdentifiers()
    result = di.combine(frozenset({"a", "b"}), frozenset({"c", "d"}))
    assert result == frozenset({"a", "b", "c", "d"})


def test_delete_identifiers_patches_combine_overlap():
    di = DeleteIdentifiers()
    result = di.combine(frozenset({"a", "b"}), frozenset({"b", "c"}))
    assert result == frozenset({"a", "b", "c"})


def test_delete_identifiers_patches_apply_simple():
    di = DeleteIdentifiers()
    target = {"a": 1, "b": 2, "c": 3}
    result = di.apply(frozenset({"b"}), target)
    assert result == {"a": 1, "c": 3}


def test_delete_identifiers_patches_apply_nested():
    di = DeleteIdentifiers()
    target = {"outer": {"a": 1, "b": 2}}
    result = di.apply(frozenset({"a"}), target)
    assert result == {"outer": {"b": 2}}


def test_delete_identifiers_patches_apply_list():
    di = DeleteIdentifiers()
    target = [{"a": 1, "b": 2}, {"a": 3, "c": 4}]
    result = di.apply(frozenset({"a"}), target)
    assert result == [{"b": 2}, {"c": 4}]


def test_delete_identifiers_patches_apply_deep():
    di = DeleteIdentifiers()
    target = {"l1": {"l2": {"l3": {"target": 1, "keep": 2}}}}
    result = di.apply(frozenset({"target"}), target)
    assert result == {"l1": {"l2": {"l3": {"keep": 2}}}}


def test_delete_identifiers_patches_apply_no_match():
    di = DeleteIdentifiers()
    target = {"a": 1, "b": 2}
    result = di.apply(frozenset({"x"}), target)
    assert result == {"a": 1, "b": 2}


def test_delete_identifiers_patches_apply_empty():
    di = DeleteIdentifiers()
    target = {"a": 1, "b": 2}
    result = di.apply(frozenset(), target)
    assert result == {"a": 1, "b": 2}


def test_delete_identifiers_patches_doesnt_modify_original():
    di = DeleteIdentifiers()
    target = {"a": 1, "b": 2}
    original = target.copy()
    _ = di.apply(frozenset({"a"}), target)
    assert target == original


def test_delete_identifiers_patches_size():
    di = DeleteIdentifiers()
    assert di.size(frozenset({"a", "b", "c"})) == 3
    assert di.size(frozenset()) == 0


# =============================================================================
# delete_identifiers pass integration tests
# =============================================================================


def json_problem(initial, is_interesting):
    return BasicReductionProblem(
        initial,
        is_interesting,
        work=WorkContext(parallelism=1),
        sort_key=lambda s: len(json.dumps(s)),
    )


async def test_can_remove_some_identifiers():
    initial = [{"a": 0, "b": 0, "c": i} for i in range(10)]

    async def is_interesting(ls):
        return len(ls) == 10 and all(x.get("c", -1) == i for i, x in enumerate(ls))

    assert await is_interesting(initial)

    problem = json_problem(initial, is_interesting)

    await delete_identifiers(problem)

    assert problem.current_test_case == [{"c": i} for i in range(10)]


async def test_delete_identifiers_all_removable():
    initial = {"a": 1, "b": 2, "c": 3}

    async def is_interesting(d):
        return isinstance(d, dict)

    problem = json_problem(initial, is_interesting)
    await delete_identifiers(problem)
    assert problem.current_test_case == {}


async def test_delete_identifiers_none_removable():
    initial = {"required": 42}

    async def is_interesting(d):
        return d.get("required") == 42

    problem = json_problem(initial, is_interesting)
    await delete_identifiers(problem)
    assert problem.current_test_case == {"required": 42}
