"""JSON-lines output shared by the three independent programs."""

import json
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Protocol, TextIO


def _json_default(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def to_json(result: object) -> str:
    """Serialize dataclasses/enums; reject nonstandard NaN/Infinity values."""
    return json.dumps(result, default=_json_default, allow_nan=False)


class Publisher(Protocol):
    def publish(self, result: object) -> None: ...


class ConsolePublisher:
    """One JSON object per line on stdout; logging stays on stderr."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream

    def publish(self, result: object) -> None:
        print(to_json(result), file=self.stream if self.stream is not None else sys.stdout, flush=True)
