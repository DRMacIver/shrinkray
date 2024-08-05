import json


def is_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


JSON_PASSES = []
