import json
from copy import deepcopy
from typing import Any

from attrs import define

from shrinkray.passes.definitions import Format, ParseError, compose
from shrinkray.passes.patching import Patches, apply_patches
from shrinkray.problem import ReductionProblem


def is_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except ValueError:
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
            return json.loads(input)
        except json.JSONDecodeError as e:
            raise ParseError(*e.args)

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
    identifiers = gather_identifiers(problem.current_test_case)

    await apply_patches(
        problem, DeleteIdentifiers(), [frozenset({id}) for id in identifiers]
    )


JSON_PASSES = [compose(JSON, delete_identifiers)]
