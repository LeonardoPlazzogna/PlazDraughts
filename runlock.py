"""
Exclusive lock on the weights folder: one run at a time.

It comes from real damage. Two conductors left running in parallel on the same
`weights/` folder both write `metrics.csv`, `train_state.pkl`, `champion.pt`,
`ladder.json` and the same temporary files of the C++ engine: the CSV ends up
with the rows of the two runs mixed, one Elo ladder overwrites the other, and
the two fight over the GPU. Nothing shouts, so it can go on for hours.

WHY AN OPERATING-SYSTEM LOCK AND NOT A PID FILE. A file holding a PID requires
deciding whether that process is still alive, and there is no portable answer:
on Windows `os.kill(pid, 0)` is not a harmless probe as on POSIX, it calls
TerminateProcess and KILLS the process. A badly written liveness check would
have been worse than the problem it solves. A heartbeat on the file's timestamp
would avoid os.kill, but a cycle lasts ~7 minutes: the staleness threshold would
have to be so high that every crash would mean waiting half an hour.

The operating-system lock has none of these problems: the kernel releases it
when the process ends, however it ends -- even if killed outright, even if the
pod is shut down. No heuristics, no ghost locks to clean up by hand.

Usage:

    with acquire(weights_dir):
        ...   # we are the only ones here

Raises RunLockError if another run already holds the folder. The message says
who holds it (PID, machine, start time), because "folder busy" without saying
by whom leaves one guessing.
"""
from __future__ import annotations

import contextlib
import json
import os
import socket
import time
from typing import Iterator, TextIO

LOCK_NAME = ".run.lock"


class RunLockError(RuntimeError):
    """The weights folder is already in use by another run."""


# --- lock primitives, one per platform ---------------------------------------
# Both are NON-blocking: the aim is to fail at once with a useful message, not
# to hang waiting for the other run to finish six hours later.
try:                                    # POSIX (Linux, macOS)
    import fcntl

    def _try_lock(fh: TextIO) -> bool:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(fh: TextIO) -> None:
        with contextlib.suppress(OSError):
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

except ImportError:                     # Windows
    import msvcrt

    # Windows locks BYTE RANGES, and the lock is mandatory: nobody can even
    # READ the locked bytes. Locking byte 0 -- the obvious choice -- made the
    # file unreadable even for whoever only wanted to know who holds the lock,
    # and the error message degraded to "unknown process", losing the only
    # useful information.
    #
    # So a byte at a very high offset is locked, far beyond the data (a few
    # dozen bytes). Mutual exclusion stays perfect -- every contender asks for
    # the same byte -- and the JSON at the start of the file stays readable by
    # anyone. Locking beyond the end of the file is legitimate on Windows and
    # does not make it grow.
    _LOCK_OFFSET = 1 << 30

    def _try_lock(fh: TextIO) -> bool:
        try:
            fh.seek(_LOCK_OFFSET)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(fh: TextIO) -> None:
        with contextlib.suppress(OSError):
            fh.seek(_LOCK_OFFSET)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)


def _read_holder(path: str) -> str:
    """Who holds the lock, in readable form. It must never raise: it is called
    while already failing, and an error here would replace a useful message
    with a useless one."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return (f"PID {d.get('pid', '?')} on {d.get('host', '?')}, "
                f"started at {d.get('started', '?')}")
    except Exception:
        return "unknown process (the lock file is not readable)"


@contextlib.contextmanager
def acquire(weights_dir: str) -> Iterator[str]:
    """Takes the exclusive lock on `weights_dir` for the duration of the block."""
    os.makedirs(weights_dir, exist_ok=True)
    path = os.path.join(weights_dir, LOCK_NAME)

    # "a+" and not "w": opening for writing TRUNCATES the file, and truncating it
    # before knowing whether the lock is ours would erase the data of the
    # rightful owner -- who could then no longer tell anyone who it is.
    fh = open(path, "a+", encoding="utf-8")
    if not _try_lock(fh):
        holder = _read_holder(path)
        fh.close()
        raise RunLockError(
            f"the weights folder '{weights_dir}' is already in use by another "
            f"run ({holder}).\n"
            "      Two runs on the same folder overwrite each other's weights, "
            "state and metrics.\n"
            "      Stop the other run, or use a different folder: "
            "DAMA_WEIGHTS_DIR=weights2 ...")

    try:
        fh.seek(0)
        fh.truncate()
        json.dump({"pid": os.getpid(), "host": socket.gethostname(),
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, fh)
        fh.flush()
        yield path
    finally:
        _unlock(fh)
        fh.close()
        # The file stays on disk: removing it would open a race in which another
        # process has already opened the same path and ends up locking an
        # unlinked inode, believing it holds the lock. An empty file of a few
        # bytes bothers nobody.
