from __future__ import annotations

import datetime as dt
from typing import Any

from msgspec import Struct


class Event(Struct):
    event: str
    timestamp: dt.datetime
    details: dict[str, Any]
