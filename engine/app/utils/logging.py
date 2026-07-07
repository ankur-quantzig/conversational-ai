from __future__ import annotations

import json


def dump_json(data) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)

