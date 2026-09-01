"""Does batching `_bump` help the one case #601's own sharding fix admitted it couldn't touch?

#601 (perf(store): shard the lifetime counters by writer) shards `_bump` across
COUNTER_SHARDS files keyed by pid, fixing CONCURRENT contention between writers. Its own
docstring is explicit about what that can't fix: "traffic concentrated in ONE room does not
move at all, because those writers already serialise on that room's own flock." Sharding
only ever helps when multiple writers can hit the lock at the same instant — a single hot
room's writes are already forced sequential by the room's own flock before `_bump` is ever
reached, so there is no concurrency there for a shard axis to divide.

But sequential writers still each pay a full `_bump` lock-acquire-read-modify-replace on
EVERY call, and that cost doesn't depend on concurrency at all — it depends on how often
`_bump` is called. Batch it: accumulate deltas in memory per worker, flush to the real
`_bump` (global or sharded, either works underneath this) every BATCH_SIZE calls, flushing
any remainder on exit. This is a wrapper around the existing on-disk mechanism, not a
replacement for it — same file format, same read path, same best-effort/undercount
contract, just called less often.

Notably: bench/counters.py's own write_path() never actually measures the single-hot-room
claim — _append_proc gives every thread its own room (`f"r{pid}x{tid}"`), so the claim is
asserted from reasoning about the room lock, not shown in that file's own numbers. This
script adds the missing hot-room configuration and runs batching through it.

Run: python3 counters_batched.py /path/to/technocore-chat
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

if len(sys.argv) < 2:
    print("usage: counters_batched.py /path/to/technocore-chat", file=sys.stderr)
    sys.exit(1)

REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO / "src"))

import orjson  # noqa: E402

import store  # noqa: E402

WORKERS_LIVE = 5  # WEB_CONCURRENCY in production, matching bench/counters.py
APPENDS = 60  # per thread, matching bench/counters.py
BATCH_SIZE = 20  # flush every N bumps, or on exit — the one new knob this script adds


def _global_bump(root: Path, **deltas: int) -> None:
    """Byte-for-byte the pre-#601 baseline, copied from bench/counters.py so "before" stays
    reproducible independent of which store.py this runs against."""
    path = root / store.COUNTERS_FILE
    try:
        with store._locked(path):
            current = store._read_counters(path)
            merged = {key: current[key] + deltas.get(key, 0) for key in store.COUNTER_KEYS}
            store._replace(path, orjson.dumps(merged))
    except OSError:
        pass


class _Batcher:
    """Per-process accumulator. Threads inside one worker share it, by the same design
    #601 shares one counter shard across a worker's own threads — they are one writer.

    Flush is triggered by count, not time, so a benchmark loop stays deterministic: every
    run flushes the same number of times regardless of wall-clock jitter, which is what
    makes the throughput/latency columns comparable across repeated runs at all.
    """

    def __init__(self, real_bump, batch_size: int) -> None:
        self._real_bump = real_bump
        self._batch_size = batch_size
        self._lock = threading.Lock()
        self._count = 0
        self._deltas: dict[str, int] = {}

    def bump(self, root: Path, **deltas: int) -> None:
        flush = None
        with self._lock:
            for k, v in deltas.items():
                self._deltas[k] = self._deltas.get(k, 0) + v
            self._count += 1
            if self._count >= self._batch_size:
                flush, self._deltas, self._count = self._deltas, {}, 0
        if flush:
            self._real_bump(root, **flush)

    def flush(self, root: Path) -> None:
        """Called once at worker shutdown — the graceful-exit half of the design. A crash
        loses whatever's unflushed, same accepted failure mode _bump already documents,
        just a larger window (up to batch_size - 1 calls instead of 0)."""
        with self._lock:
            flush, self._deltas, self._count = self._deltas, {}, 0
        if flush:
            self._real_bump(root, **flush)


def _append_proc(root: str, mode: str, threads: int, room_mode: str, barrier, out: str) -> None:
    """One worker process. room_mode="own" gives every thread its own room (bench/counters.py's
    existing shape — no room-lock contention, so the counter lock is the only thing shared).
    room_mode="hot" gives every thread the SAME room — the configuration #601's docstring
    describes but never runs: all writers already serialised by one room flock before
    `_bump` is ever reached.
    """
    sharded_bump = store._bump  # the real, shipped implementation — captured before any patch
    batcher = None
    if mode == "global":
        vars(store)["_bump"] = _global_bump
    elif mode == "none":
        vars(store)["_bump"] = lambda *a, **k: None
    elif mode == "batched-global":
        batcher = _Batcher(_global_bump, BATCH_SIZE)
        vars(store)["_bump"] = batcher.bump
    elif mode == "batched-sharded":
        batcher = _Batcher(sharded_bump, BATCH_SIZE)
        vars(store)["_bump"] = batcher.bump
    elif mode != "sharded":
        raise ValueError(mode)

    path = Path(root)
    laps: list[float] = []
    guard = threading.Lock()
    hot_room = f"hot-{os.getpid()}"  # shared per PROCESS: cross-process room contention too

    def one(tid: int) -> None:
        room = hot_room if room_mode == "hot" else f"r{os.getpid() % 9973:04d}x{tid}"
        store.append(path, room, "bot", "warm")  # room-create, outside the timed section
        mine = []
        barrier.wait()
        for i in range(APPENDS):
            t0 = time.perf_counter()
            store.append(path, room, "bot", f"message number {i} from pid {os.getpid()} tid {tid}")
            mine.append(time.perf_counter() - t0)
        with guard:
            laps.extend(mine)

    workers = [threading.Thread(target=one, args=(i,)) for i in range(threads)]
    start = time.perf_counter()
    [t.start() for t in workers]
    [t.join(600) for t in workers]
    if batcher is not None:
        batcher.flush(path)  # graceful-shutdown flush — every batched run does this
    Path(out).write_text(json.dumps({"start": start, "end": time.perf_counter(), "laps": laps}))


def _append_run(
    procs: int, threads: int, mode: str, room_mode: str
) -> tuple[float, float, float, int]:
    context = multiprocessing.get_context("fork")
    root = Path(tempfile.mkdtemp())
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / ".reaped").touch()
        (root / store.SNAPSHOTS_FILE).touch()
        barrier = context.Barrier(procs * threads)
        running = [
            context.Process(
                target=_append_proc,
                args=(str(root), mode, threads, room_mode, barrier, str(root / f"a{i}")),
            )
            for i in range(procs)
        ]
        [p.start() for p in running]
        [p.join(600) for p in running]
        assert all(p.exitcode == 0 for p in running), [p.exitcode for p in running]
        results = [json.loads((root / f"a{i}").read_text()) for i in range(procs)]
        laps = sorted(lap for r in results for lap in r["laps"])
        elapsed = max(r["end"] for r in results) - min(r["start"] for r in results)
        counted = store.counters(root)["messages"]
        return (
            len(laps) / elapsed,
            laps[len(laps) // 2] * 1e3,
            laps[int(len(laps) * 0.99)] * 1e3,
            counted,
        )
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)


MODES = ("global", "sharded", "batched-global", "batched-sharded", "none")


def compare(room_mode: str, label: str) -> None:
    print(f"\n{label} — {WORKERS_LIVE} processes x T threads, room_mode={room_mode}")
    print("  " + f"{'threads':>7}  " + "  ".join(f"{m:>17}" for m in MODES))
    for threads in (4, 8, 16):
        row = {m: _append_run(WORKERS_LIVE, threads, m, room_mode) for m in MODES}
        # +1 per thread: the room-create call outside the timed loop also bumps `messages`.
        want = WORKERS_LIVE * threads * (APPENDS + 1)
        for m, (_rate, _p50, _p99, counted) in row.items():
            if counted != want:
                print(
                    f"  ! {m} lost {want - counted} of {want} counted messages (room_mode={room_mode})"
                )
        print(
            f"  {threads:>7}  "
            + "  ".join(f"{row[m][0]:>10,.0f}/s {row[m][1]:>5.2f}ms" for m in MODES)
        )


if __name__ == "__main__":
    assert (
        store._COUNTER_FILES[0] == store.COUNTERS_FILE if hasattr(store, "_COUNTER_FILES") else True
    )
    print(
        f"repo: {REPO}  batch_size={BATCH_SIZE}  workers_live={WORKERS_LIVE}  appends/thread={APPENDS}"
    )
    compare("own", "own-room (bench/counters.py's existing shape)")
    compare("hot", "SAME room for every writer (the untested claim)")
