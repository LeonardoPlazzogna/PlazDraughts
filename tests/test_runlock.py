"""
Tests of the exclusive lock on the weights folder (runlock.py).

The test that matters is the fourth: the lock must be released when the process
holding it is KILLED OUTRIGHT, without being able to run any cleanup. It is the
real scenario -- the OOM killer, a pod switched off -- and the only reason an
operating-system lock is used instead of a file with the PID inside. If this
test did not pass, after every crash the folder would stay locked forever.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runlock import acquire, RunLockError, LOCK_NAME     # noqa: E402


def test_exclusive_lock_in_same_process():
    with tempfile.TemporaryDirectory() as d:
        with acquire(d):
            try:
                with acquire(d):
                    raise AssertionError("the second lock was NOT supposed to succeed")
            except RunLockError as e:
                assert "already in use" in str(e)
                assert str(os.getpid()) in str(e), \
                    "the message must say WHO holds the lock"
    print("  ok  two locks on the same folder: the second is rejected")


def test_release_at_end_of_block():
    with tempfile.TemporaryDirectory() as d:
        with acquire(d):
            pass
        with acquire(d):            # must succeed: the first one released it
            pass
    print("  ok  after leaving the block the lock is free")


def test_different_folders_do_not_interfere():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        with acquire(d1), acquire(d2):
            pass
    print("  ok  different folders: independent locks")


def _child(d: str) -> str:
    return textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})
        from runlock import acquire
        with acquire({d!r}):
            print("TAKEN", flush=True)
            time.sleep(120)
    """)


def test_lock_across_processes_and_release_after_kill():
    """The case that justifies the whole design choice."""
    with tempfile.TemporaryDirectory() as d:
        p = subprocess.Popen([sys.executable, "-c", _child(d)],
                             stdout=subprocess.PIPE, text=True)
        try:
            assert p.stdout.readline().strip() == "TAKEN", \
                "the child did not take the lock"

            # 1) while the child is alive, we must NOT be able to get in
            try:
                with acquire(d):
                    raise AssertionError(
                        "we took the lock while another process was holding it")
            except RunLockError as e:
                assert str(p.pid) in str(e), \
                    f"the message must report the PID {p.pid} of the holder"
            print("  ok  another process holds the lock: we are rejected")

            # 2) OUTRIGHT kill: no cleanup possible on the child's side
            p.kill()
            p.wait(timeout=10)
        finally:
            if p.poll() is None:
                p.kill(); p.wait(timeout=10)

        # 3) the lock must be free, even though the file is still there
        assert os.path.exists(os.path.join(d, LOCK_NAME))
        deadline = time.time() + 10
        while True:
            try:
                with acquire(d):
                    break
            except RunLockError:
                if time.time() > deadline:
                    raise AssertionError(
                        "lock still taken after the holder was killed: "
                        "exactly the defect this mechanism must prevent")
                time.sleep(0.2)
        print("  ok  holder killed outright -> lock released by the operating system")


def main():
    print("[test] exclusive lock on the weights folder")
    test_exclusive_lock_in_same_process()
    test_release_at_end_of_block()
    test_different_folders_do_not_interfere()
    test_lock_across_processes_and_release_after_kill()
    print("[test] runlock: all ok")


if __name__ == "__main__":
    main()
