from __future__ import annotations

import struct
import threading
import zlib
from collections import deque
from pathlib import Path

import msgspec

from app.observability.events import ObservabilityEvent

MAGIC = b"OBSV1"
HEADER = struct.Struct(">II")


class RingEventStorage:
    def __init__(self, path: Path, max_bytes: int) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_bytes(b"")

    def append(self, event: ObservabilityEvent) -> None:
        data = msgspec.msgpack.encode(event)
        record = MAGIC + HEADER.pack(len(data), zlib.crc32(data)) + data
        with self._lock:
            with self.path.open("ab") as fh:
                fh.write(record)
                fh.flush()
            if self.path.stat().st_size > self.max_bytes:
                self._compact_locked()

    def read_latest(self, limit: int) -> list[ObservabilityEvent]:
        limit = max(0, limit)
        if limit == 0:
            return []
        with self._lock:
            events = deque(maxlen=limit)
            for event in self._iter_file_locked():
                events.append(event)
            return list(events)

    def replay(self, limit: int | None = None) -> list[ObservabilityEvent]:
        maxlen = None if limit is None else max(0, limit)
        with self._lock:
            if maxlen is None:
                return list(self._iter_file_locked())
            events = deque(maxlen=maxlen)
            for event in self._iter_file_locked():
                events.append(event)
            return list(events)

    def _compact_locked(self) -> None:
        records: deque[bytes] = deque()
        total = 0
        for raw_record, _event in self._iter_raw_records_locked():
            records.append(raw_record)
            total += len(raw_record)
            while total > self.max_bytes and records:
                total -= len(records.popleft())
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("wb") as fh:
            for record in records:
                fh.write(record)
        tmp_path.replace(self.path)

    def _iter_file_locked(self):
        for _raw_record, event in self._iter_raw_records_locked():
            yield event

    def _iter_raw_records_locked(self):
        with self.path.open("rb") as fh:
            while True:
                magic = fh.read(len(MAGIC))
                if not magic:
                    return
                if magic != MAGIC:
                    return
                header = fh.read(HEADER.size)
                if len(header) != HEADER.size:
                    return
                size, checksum = HEADER.unpack(header)
                data = fh.read(size)
                if len(data) != size:
                    return
                if zlib.crc32(data) != checksum:
                    continue
                try:
                    event = msgspec.msgpack.decode(data, type=ObservabilityEvent)
                except msgspec.DecodeError:
                    continue
                yield magic + header + data, event
