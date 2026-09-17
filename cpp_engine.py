"""
Bridge between the Python conductor and the C++ self-play engine (engine_c/).

The C++ engine plays games much faster (compiled move generation, MCTS and
encoding, threads instead of processes, large inference batches without IPC),
but it does NOT train: it replaces only the self-play and arena games. Training
and the promotion decision stay in PyTorch/Python, unchanged.

Per-cycle flow, when the engine is enabled:
    champion (state_dict) -> TorchScript -> dama_engine (N threads, GPU)
        -> binary dataset -> cpp_dataset.load_dataset() -> replay buffer

The C++ search hyperparameters (c_puct 1.5, temp_moves 12, Dirichlet 1.5/0.25,
max_plies 300) are the same as in the Python self-play: the two paths produce
samples with the SAME exploration, so the buffer stays homogeneous even when
they alternate.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import tempfile

import torch

from cpp_dataset import load_dataset


class CppEngineError(RuntimeError):
    pass


def export_jit(net, out_path: str, in_planes: int) -> str:
    """Saves `net` as TorchScript in the format the C++ engine loads.

    It is traced and frozen on CPU: the C++ backend loads the model on the
    requested device (libtorch_backend.hpp), so there is no need to export one
    version per device. The freeze folds BatchNorm into the convolutions, as the
    Python evaluator does.
    """
    was_training = net.training
    net.eval()
    try:
        example = torch.zeros(4, in_planes, 8, 8)
        with torch.no_grad():
            traced = torch.jit.freeze(torch.jit.trace(net.cpu(), example))
        tmp = out_path + ".tmp"
        traced.save(tmp)
        os.replace(tmp, out_path)      # atomic: never a half-written file
    finally:
        if was_training:
            net.train()
    return out_path


def selfplay(engine_bin: str, net, *, n_games: int, n_sims: int, n_threads: int,
             device: str = "cuda", fp16: bool = False, mcts_batch: int = 8,
             max_batch: int = 256, seed: int = 1, in_planes: int = 7,
             c_puct: float = 1.5, value_discount: float = 1.0,
             workdir: str | None = None, timeout: float | None = None):
    """Plays `n_games` games with the C++ engine and returns (samples, stdout),
    with the samples in the same form as the Python self-play: a list of
    (X, pi, mask, z).

    `net=None` runs the engine WITHOUT a network (the engine's internal fake
    backend): the samples have no playing value, it only validates the pipeline
    -- invocation, dataset format, reading back -- on a machine without
    LibTorch.

    Raises CppEngineError if the engine fails, so the caller can decide to fall
    back on the Python path instead of stopping the run.
    """
    if not os.path.exists(engine_bin):
        raise CppEngineError(f"executable not found: {engine_bin}")
    # On Windows a RELATIVE path passes the check above but cannot be started:
    # CreateProcess does not resolve it the way os.path does, and the error
    # that comes back ("file not found") contradicts the check just passed.
    # Verified: with the absolute path the same executable starts. It is
    # normalized here, once, for every caller.
    engine_bin = os.path.abspath(engine_bin)

    # If we create the folder, we must remove it. A caller that passes one owns
    # it and manages it: the conductor reuses its own for the whole run, and
    # deleting it under its feet would throw away the exported model that the
    # arena reuses between blocks. Without this distinction every call without
    # workdir leaves behind a folder with the exported models inside, and they
    # pile up quickly.
    owned = workdir is None
    tmpdir = workdir or tempfile.mkdtemp(prefix="dama_cpp_")
    os.makedirs(tmpdir, exist_ok=True)
    try:
        jit_path = os.path.join(tmpdir, "champion_jit.pt")
        out_path = os.path.join(tmpdir, "selfplay.bin")

        cmd = [engine_bin]
        if net is not None:
            export_jit(net, jit_path, in_planes)
            cmd += ["--weights", jit_path]
        cmd += ["--device", device,
               "--n-games", str(n_games),
               "--n-threads", str(n_threads),
               "--n-sims", str(n_sims),
               "--mcts-batch", str(mcts_batch),
               "--max-batch", str(max_batch),
               "--seed", str(seed),
               "--c-puct", str(c_puct),
               "--out", out_path]
        # Only if different from 1: an engine built before the option existed
        # keeps working as long as the discount stays off, instead of dying with
        # "unknown option" on a configuration that does not use it.
        if value_discount != 1.0:
            cmd += ["--value-discount", str(value_discount)]
        if fp16:
            cmd.append("--fp16")

        try:
            # An explicit encoding, and it is not pedantry. With text=True alone,
            # Python decodes with the LOCALE encoding: cp1252 on Windows, ASCII if
            # the locale is "C". A single byte outside that repertoire -- an
            # accent in a message, or garbage from a dying process -- and
            # decoding fails.
            #
            # And it fails in the worst way: the exception is raised in the
            # THREAD that reads the pipe, so subprocess.run does NOT propagate it
            # and returns with stdout set to None. The code below would carry on
            # as if the engine had said nothing, and the error would show up
            # later, disguised and far from its cause. Verified: it really
            # happens.
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise CppEngineError(f"the C++ engine did not finish within {timeout}s") from e

        if proc.returncode != 0:
            raise CppEngineError(
                f"the C++ engine exited with code {proc.returncode}.\n"
                f"command: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

        if not os.path.exists(out_path):
            raise CppEngineError(f"the engine did not produce {out_path}\n{proc.stdout}")

        samples = load_dataset(out_path)
        try:
            os.remove(out_path)
        except OSError:
            pass
        return samples, proc.stdout
    finally:
        # The folder we created goes away IN ANY CASE, even if the engine failed
        # -- and it is exactly when it fails that one goes looking at what is
        # left on disk. With the cleanup only on the success path, EMPTY folders
        # remained: created, then abandoned by an exception raised before
        # anything was written. The CALLER's folder is not touched: the
        # conductor keeps there the exported model that the arena reuses
        # between blocks.
        if owned:
            shutil.rmtree(tmpdir, ignore_errors=True)


def parse_stats(engine_stdout: str) -> dict:
    """Extracts the "STATS key=value ..." line printed by the engine.

    It is how the C++ path fills the same metrics.csv columns that the Python
    path fills on its own: without it, when switching to the C++ engine the
    diagnostic metrics would disappear from the CSV exactly in the fastest
    configuration, i.e. the one used for the real run.
    """
    for line in engine_stdout.splitlines():
        if not line.startswith("STATS "):
            continue
        out = {}
        for tok in line[len("STATS "):].split():
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            try:
                out[k] = float(v) if "." in v else int(v)
            except ValueError:
                pass
        return out
    return {}


def parse_hub_batches(engine_stdout: str) -> list[float]:
    """Mean size of the batches sent to the network, one value per hub.

    Self-play has a single hub; the arena has TWO, because the two networks
    differ and cannot share a batch. It is the number that decides how well the
    card is used: the larger the batch, the fewer passes and the more useful work
    per pass. It is for tuning, not for the run.

    Returns an empty list with engines built before the figure existed.
    """
    out = []
    for line in engine_stdout.splitlines():
        if line.strip().startswith(("hub batches", "hub A batches", "hub B batches")):
            for tok in line.replace("(", " ").replace(")", " ").split():
                try:
                    v = float(tok)
                except ValueError:
                    continue
                if "." in tok:          # the mean, not the batch count
                    out.append(v)
                    break
    return out


def parse_arena_distinct(engine_stdout: str) -> int | None:
    """Number of games with a distinct move sequence, if the engine reports it.

    It is the sample size that really counts: two identical games carry the
    information of one, so a confidence interval computed on the number of games
    played would be narrower than it really is. Returns None with engines built
    before this figure existed.
    """
    for line in engine_stdout.splitlines():
        if line.startswith("ARENA_SCORE "):
            for tok in line.split()[2:]:
                if tok.startswith("distinct="):
                    try:
                        return int(tok.split("=", 1)[1])
                    except ValueError:
                        return None
    return None


def arena(engine_bin: str, net_a, net_b, *, n_games: int, n_sims: int,
          n_threads: int, device: str = "cuda", fp16: bool = False,
          mcts_batch: int = 8, max_batch: int = 256, seed: int = 1,
          in_planes: int = 7, c_puct: float = 1.5,
          c_puct_b: float | None = None, n_sims_b: int | None = None,
          mcts_batch_b: int | None = None,
          temp_plies: int | None = None,
          workdir: str | None = None,
          reuse_jit: bool = False,
          timeout: float | None = None):
    """Plays `net_a` against `net_b` with the C++ engine and returns A's score in
    [0,1] (1 = A always wins), A's (wins, draws, losses) and the engine's stdout.

    As for `selfplay`, `net_a=net_b=None` runs the engine with the fake backend:
    it validates the connection without LibTorch. Raises CppEngineError on
    problems, so the caller can fall back on the Python arena.
    """
    if not os.path.exists(engine_bin):
        raise CppEngineError(f"executable not found: {engine_bin}")
    # On Windows a RELATIVE path passes the check above but cannot be started:
    # CreateProcess does not resolve it the way os.path does, and the error
    # that comes back ("file not found") contradicts the check just passed.
    # Verified: with the absolute path the same executable starts. It is
    # normalized here, once, for every caller.
    engine_bin = os.path.abspath(engine_bin)

    # If we create the folder, we must remove it. A caller that passes one owns
    # it and manages it: the conductor reuses its own for the whole run, and
    # deleting it under its feet would throw away the exported model that the
    # arena reuses between blocks. Without this distinction every call without
    # workdir leaves behind a folder with the exported models inside, and they
    # pile up quickly.
    owned = workdir is None
    tmpdir = workdir or tempfile.mkdtemp(prefix="dama_cpp_")
    os.makedirs(tmpdir, exist_ok=True)
    try:

        cmd = [engine_bin, "--mode", "arena"]
        if net_a is not None and net_b is not None:
            jit_a = os.path.join(tmpdir, "arena_a.pt")
            jit_b = os.path.join(tmpdir, "arena_b.pt")
            # REUSE if the models were already exported and the networks have
            # not changed. The sequential-test arena calls this function once per
            # BLOCK, several times per cycle, and the two networks stay the same
            # for the whole match: exporting them again every time means paying
            # for the TorchScript trace once per block instead of once per
            # match.
            if not reuse_jit or not (os.path.exists(jit_a) and os.path.exists(jit_b)):
                export_jit(net_a, jit_a, in_planes)
                export_jit(net_b, jit_b, in_planes)
            cmd += ["--weights", jit_a, "--weights-b", jit_b]
        cmd += ["--device", device,
                "--n-games", str(n_games),
                "--n-threads", str(n_threads),
                "--n-sims", str(n_sims),
                "--mcts-batch", str(mcts_batch),
                "--max-batch", str(max_batch),
                "--seed", str(seed),
                "--c-puct", str(c_puct)]
        # Different exploration for side B: it measures the SEARCH instead of
        # the network -- the very same network on both sides and two different
        # c_puct values, so whoever wins has the better search. None = same value
        # as A, the normal case of the promotion arena.
        if c_puct_b is not None:
            cmd += ["--c-puct-b", str(c_puct_b)]
        # Different simulations for side B: it answers the budget question --
        # whether the extra search is worth the games it costs. None = same as A.
        if n_sims_b is not None:
            cmd += ["--n-sims-b", str(n_sims_b)]

        # Different leaf batching between the two sides: with the same network it
        # is the only way to know whether more batching makes the search worse.
        if mcts_batch_b is not None:
            cmd += ["--mcts-batch-b", str(mcts_batch_b)]
        # Sampled opening plies. It matters more than it seems: without
        # Dirichlet noise and with deterministic networks, two games that share
        # the opening are THE SAME GAME. The distinct games are at most as many
        # as the distinct openings, and a confidence interval computed on the
        # number of games played -- instead of the really different ones -- is
        # too narrow. None = the engine default (4); the conductor passes
        # DAMA_ARENA_TEMP_PLIES (10 by default), and calibrating the search,
        # where the two networks are identical, needs it raised.
        if temp_plies is not None:
            cmd += ["--temp-plies", str(temp_plies)]
        if fp16:
            cmd.append("--fp16")

        try:
            # explicit encoding: see the longer note in selfplay(). In short,
            # without it a byte that cannot be decoded in the locale encoding
            # silently loses ALL of the arena output, ARENA_SCORE included.
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise CppEngineError(f"the C++ arena did not finish within {timeout}s") from e

        if proc.returncode != 0:
            raise CppEngineError(
                f"the C++ arena exited with code {proc.returncode}.\n"
                f"command: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

        # The engine prints a dedicated "ARENA_SCORE <value>" line: that one is
        # parsed and not the last line, so any extra messages from the engine do
        # not move the result.
        score, wdl = None, None
        for line in proc.stdout.splitlines():
            if line.startswith("ARENA_SCORE "):
                try:
                    parts = line.split()
                    score = float(parts[1])
                    # counts (for the SPRT): "wins=N draws=N losses=N"
                    kv = dict(t.split("=", 1) for t in parts[2:] if "=" in t)
                    if {"wins", "draws", "losses"} <= kv.keys():
                        wdl = (int(kv["wins"]), int(kv["draws"]), int(kv["losses"]))
                except (IndexError, ValueError):
                    pass
        if score is None:
            raise CppEngineError(
                f"the C++ arena did not report ARENA_SCORE.\nstdout:\n{proc.stdout}")
        if not (0.0 <= score <= 1.0):
            raise CppEngineError(f"arena score outside [0,1]: {score}")
        if wdl is None:
            # older engine without counts: they are rebuilt from the score. A
            # coarse approximation (no draws), but the caller does not break; the
            # SPRT will only be more conservative than needed.
            w = int(round(score * n_games))
            wdl = (w, 0, n_games - w)
        return score, wdl, proc.stdout
    finally:
        # The folder we created goes away IN ANY CASE, even if the engine failed
        # -- and it is exactly when it fails that one goes looking at what is
        # left on disk. With the cleanup only on the success path, EMPTY folders
        # remained: created, then abandoned by an exception raised before
        # anything was written. The CALLER's folder is not touched: the
        # conductor keeps there the exported model that the arena reuses
        # between blocks.
        if owned:
            shutil.rmtree(tmpdir, ignore_errors=True)
