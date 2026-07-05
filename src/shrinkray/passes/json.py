import json
from copy import deepcopy
from typing import Any

from attrs import define

from shrinkray.passes.definitions import ReductionPass
from shrinkray.passes.patching import Patches, apply_patches
from shrinkray.problem import Format, ParseError, ReductionProblem


# The JSON passes copy and walk the parsed structure (e.g. deepcopy in
# DeleteIdentifiers.apply), which recurses per nesting level and blows
# Python's recursion limit a few hundred levels deep. Rather than let
# that crash shrink ray, we decline to treat input nested past this bound
# as JSON, so it falls back to the generic byte-level passes (which have
# no trouble reducing deeply nested text). The bound sits above realistic
# JSON nesting and well below where the recursive passes fail (~450).
MAX_JSON_DEPTH = 256


def _exceeds_json_depth(value: Any, limit: int) -> bool:
    """Whether the nesting depth of a parsed JSON value exceeds limit.

    Iterative so that checking the depth cannot itself overflow the stack.
    """
    stack = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > limit:
            return True
        if isinstance(node, dict):
            stack.extend((child, depth + 1) for child in node.values())
        elif isinstance(node, list):
            stack.extend((child, depth + 1) for child in node)
    return False


@define(frozen=True)
class _JSON(Format[bytes, Any]):
    def __repr__(self) -> str:
        return "JSON"

    @property
    def name(self) -> str:
        return "JSON"

    def parse(self, input: bytes) -> Any:
        try:
            result = json.loads(input)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ParseError(*e.args)
        except RecursionError as e:
            # json.loads recurses per nesting level, so deeply nested input
            # overflows the stack. On Python 3.14 this surfaces as a
            # RecursionError rather than a crash; treat such input as
            # unparseable so is_valid reports it invalid instead of the
            # error escaping and taking shrink ray down.
            raise ParseError(*e.args)
        if _exceeds_json_depth(result, MAX_JSON_DEPTH):
            raise ParseError("JSON nesting too deep for structured reduction")
        return result

    def dumps(self, input: Any) -> bytes:
        return json.dumps(input).encode("utf-8")


JSON = _JSON()


def gather_identifiers(value: Any) -> set[str]:
    result = set()
    stack = [value]
    while stack:
        target = stack.pop()
        if isinstance(target, dict):
            result.update(target.keys())
            stack.extend(target.values())
        elif isinstance(target, list):
            stack.extend(target)
    return result


class DeleteIdentifiers(Patches[frozenset[str], Any]):
    @property
    def empty(self) -> frozenset[str]:
        return frozenset()

    def combine(self, *patches: frozenset[str]) -> frozenset[str]:
        result = set()
        for p in patches:
            result.update(p)
        return frozenset(result)

    def apply(self, patch: frozenset[str], target: Any) -> Any:
        target = deepcopy(target)
        stack = [target]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for k in patch:
                    value.pop(k, None)
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
        return target

    def size(self, patch: frozenset[str]) -> int:
        return len(patch)


async def delete_identifiers(problem: ReductionProblem[Any]):
    """Remove object keys from JSON structures.

    Finds all string keys used in any nested object and tries to remove
    them. When a key is removed, it's deleted from all objects that
    contain it throughout the JSON tree.
    """
    identifiers = gather_identifiers(problem.current_test_case)

    await apply_patches(
        problem, DeleteIdentifiers(), [frozenset({id}) for id in identifiers]
    )


JSON_PASSES: list[ReductionPass[Any]] = [delete_identifiers]
