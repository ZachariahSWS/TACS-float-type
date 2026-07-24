from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    if int(config.get("config_version", 0)) != 1:
        raise ValueError("Config must declare config_version = 1")
    return config


def config_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
