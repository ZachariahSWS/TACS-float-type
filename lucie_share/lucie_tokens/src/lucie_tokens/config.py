from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any


ENVIRONMENT_VARIABLE = re.compile(r"\$(?:\{([^}]+)\}|([A-Za-z_][A-Za-z0-9_]*))")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        missing = {
            braced or plain
            for braced, plain in ENVIRONMENT_VARIABLE.findall(value)
            if (braced or plain) not in os.environ
        }
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Undefined environment variables in config: {names}")
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    if int(config.get("config_version", 0)) != 1:
        raise ValueError("Config must declare config_version = 1")
    return _expand_environment(config)


def config_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
