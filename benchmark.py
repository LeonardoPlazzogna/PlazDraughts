"""How long the complete run would take on THIS machine.

    python benchmark.py                               estimate for 100 cycles
    python benchmark.py --device cuda
    python benchmark.py --weights weights/champion.pt more precise (see below)
    python benchmark.py --games 12                    larger sample

It returns an estimate, not a verdict: how much a cycle costs, how it breaks
down, and how much the whole run would cost.

WHY IT IS NOT ESTIMATED ON PAPER. Between the compiled engine on a graphics
card and the interpreted path on a processor there are almost two orders of
magnitude. Where a given machine falls inside that range can only be known by
measuring it.

WHAT IT MEASURES. A cycle has several phases and their proportion is not what
one would guess: the arena, not self-play, is usually the largest item, so a
benchmark that timed only self-play would be off by a wide factor. Here
self-play, arena, training and the fixed-depth opponents are measured
separately, and the cycle is recomposed with the real cadences (generation
ladder every 10 cycles, strength test every 20, deep anchor every 60).

HOW IT MEASURES. Through the SAME mechanisms as the run: the same process pool
and, when the network is not on the CPU, the same central inference server; with
--engine, the compiled engine's own self-play and arena. Timing a game inside a
single process would give a nicer and false number, because it would leave out
the cost of inter-process communication and the gain of batching the requests --
i.e. what decides the real speed.

THE MAIN SOURCE OF ERROR, and how it is handled. The cost of a game is
proportional to its LENGTH, and games get longer as the network learns (the
figures are in docs/results.md, section 1). A benchmark that timed the games of
a freshly initialized network would underestimate the cost. So what is measured
here is not the cost per game but the cost PER PLY -- which depends on the
machine and not on the state of training -- and it is projected onto the
steady-state length.

With --weights it starts from an already trained network and the length is
measured instead of assumed: it is the most precise variant, and the one to
prefer if a champion from an earlier run is available.
"""
from __future__ import annotations

import argparse
import io
import os
import platform
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import torch


# Steady-state game length, measured on real runs (see docs/results.md, section 1).
STEADY_PLIES = 155.0


class _Tee:
    """Sends everything to the screen AND to a file.

    It exists for the case where whoever measures is not whoever reads: the
    compute machine is elsewhere, and the results must stay on disk instead of
    in a terminal's scrollback.
    """

    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)
        # Without a flush, an interruption halfway leaves the file truncated
        # exactly where it had got to: the most interesting part.
        self.file.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False


def open_report(path: str) -> str:
    """Starts duplicating the output to a file. Returns the real path."""
    import atexit
    import datetime
    if path in ('', 'auto'):
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.abspath(f'bench_{stamp}.txt')
    else:
        path = os.path.abspath(path)
    f = io.open(path, 'w', encoding='utf-8')
    f.write(f"# benchmark, {' '.join(sys.argv)}\n")
    f.write(f"# machine: {platform.node()}  {platform.platform()}\n\n")
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)

    def close():
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        f.close()
        print(f"\n  Results saved to: {path}")

    atexit.register(close)
    return path


def _rule(c="-"):
    print(c * 70)


def _duration(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    if s < 3 * 86400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"


def _kill_tree(proc) -> None:
    """Closes a process AND its descendants.

    Windows has no process groups as Unix does: killing the parent leaves the
    children alive and orphaned, and nobody reaps them any more. It is delegated
    to taskkill, which knows how to walk the tree.
    """
    import signal
    import subprocess
    if os.name == 'nt':
        subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                       capture_output=True)
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def measure_usable_cores(repeats: int = 3) -> float | None:
    """Usable cores, asking usable_cores.py.

    It runs as a SEPARATE PROCESS and not in here: on Windows the children
    re-import the main module, and this one imports torch. Done inline, dozens of
    children import torch dozens of times and the measurement never finishes.
    """
    import subprocess
    prog = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'usable_cores.py')
    if not os.path.exists(prog):
        return None
    # The child has children of its own -- the measurement saturates the machine
    # with dozens of processes -- and at the deadline ALL of them must be closed.
    # subprocess.run with just a timeout kills the leader and leaves the
    # grandchildren alive: observed, 27 processes left around for over twenty
    # hours after a measurement gone wrong. On Windows taskkill /T is needed,
    # elsewhere the process group.
    kw = {} if os.name == 'nt' else {'start_new_session': True}
    proc = subprocess.Popen(
        [sys.executable, prog, '--repeats', str(repeats)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding='utf-8', errors='replace', **kw)
    try:
        out, err = proc.communicate(timeout=300)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        out, err = proc.communicate()
        code = proc.returncode
        print('  the core measurement did not finish within 300s: stopped')
        print('  together with all its processes. Continuing without it.')
    except Exception as e:
        print(f"  core measurement not run: {type(e).__name__}: {e}")
        return None

    class _Result:
        pass
    r = _Result()
    r.stdout, r.stderr, r.returncode = out, err, code
    for line in (r.stdout or '').splitlines():
        print('  ' + line.rstrip())
    m = re.search(r'DAMA_CPP_THREADS=(\d+)', r.stdout or '')
    if m:
        return float(m.group(1))
    # A failure of the subprocess must NOT be swallowed: without this, the
    # measurement is skipped silently and the benchmark goes on with the
    # declared cores as if nothing had happened.
    print(f"  the core measurement produced no value (exit {r.returncode}).")
    tail = (r.stderr or "").strip().splitlines()
    for line in tail[-4:]:
        print(f"    {line}")
    return None


def environment(device: str) -> None:
    _rule("=")
    print("  ENVIRONMENT")
    _rule("=")
    print(f"  python               : {sys.version.split()[0]}")
    print(f"  torch                : {torch.__version__}")
    print(f"  declared cores       : {os.cpu_count()}")
    cuda = torch.cuda.is_available()
    print(f"  CUDA available       : {cuda}")
    if cuda:
        print(f"  card                 : {torch.cuda.get_device_name(0)}")
        print(f"  torch's CUDA         : {torch.version.cuda}")
    print(f"  requested device     : {device}")

    # It is not enough for the library to declare the card available: an
    # allocation is really made. A declared but non-working device silently makes
    # everything fall back on the CPU, and the benchmark would measure another
    # configuration.
    if device != "cpu":
        try:
            torch.zeros(8, device=device) * 2
            print(f"  allocation on {device}  : succeeded")
        except Exception as e:
            print(f"\n  ERROR: '{device}' is not usable: {type(e).__name__}: {e}")
            print("  Run again with --device cpu, or fix the torch installation.")
            raise SystemExit(2)
    elif cuda:
        print("\n  WARNING: there is a usable card but you are measuring the")
        print("  CPU. Run again with --device cuda for the number that matters.")
    print()


def measure_engine(args, net_kwargs, ckpt: str) -> dict:
    """Self-play and arena from the COMPILED ENGINE, i.e. from the path the
    conductor really uses when the engine is there.

    Without this route the benchmark would measure the interpreted path -- which
    with the engine available is only the FALLBACK -- and would return a
    pessimistic estimate exactly in the best configuration. The two routes are
    not comparable: threads in a single process against separate processes,
    shared memory against inter-process queues.

    The model EXPORT is measured too, which the conductor redoes at every cycle
    to hand the network to the engine: it is a per-cycle cost like the others,
    and easy to forget because it belongs to none of the phases.
    """
    import cpp_engine
    from model import PolicyValueNet
    from train import train_on_samples

    net = PolicyValueNet(**net_kwargs)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()

    m = {}
    _rule("=")
    print(f"  1. SELF-PLAY from the compiled engine "
          f"({args.games} games, {args.sims} simulations, {args.threads} threads)")
    _rule("=")
    t0 = time.time()
    samples, out = cpp_engine.selfplay(
        args.engine, net, n_games=args.games, n_sims=args.sims,
        n_threads=args.threads, device=args.device, fp16=args.fp16,
        mcts_batch=args.mcts_batch, max_batch=args.server_batch,
        seed=12345, c_puct=1.5, value_discount=args.discount)
    dt = time.time() - t0
    st = cpp_engine.parse_stats(out)
    plies = st.get("plies_mean") or 0.0
    if not plies:
        raise SystemExit("  the engine did not report the game length")
    m["measured_plies"] = float(plies)
    m["t_ply_sp"] = dt / (args.games * plies)
    print(f"  total time             : {_duration(dt)}")
    print(f"  mean plies             : {plies:.0f}")
    print(f"  cost per ply           : {m['t_ply_sp']*1000:.2f} ms")
    print(f"  games per hour         : {3600*args.games/dt:.0f}")
    print(f"  samples                : {len(samples)}")
    print()

    _rule("=")
    print(f"  2. ARENA from the compiled engine ({args.arena} games)")
    _rule("=")
    t0 = time.time()
    cpp_engine.arena(args.engine, net, net, n_games=args.arena,
                     n_sims=args.sims, n_threads=args.threads, device=args.device,
                     fp16=args.fp16, mcts_batch=args.mcts_batch,
                     max_batch=args.server_batch, seed=999,
                     temp_plies=args.temp_plies)
    dt = time.time() - t0
    m["t_ply_ar"] = dt / (args.arena * plies)
    print(f"  total time             : {_duration(dt)}")
    print(f"  cost per ply           : {m['t_ply_ar']*1000:.2f} ms")
    print()

    _rule("=")
    print("  3. MODEL EXPORT (once per cycle)")
    _rule("=")
    import tempfile as _tf
    d = _tf.mkdtemp()
    t0 = time.time()
    cpp_engine.export_jit(net, os.path.join(d, "jit.pt"), 7)
    m["t_export"] = time.time() - t0
    shutil.rmtree(d, ignore_errors=True)
    print(f"  TorchScript export     : {_duration(m['t_export'])}")
    print()

    _rule("=")
    print(f"  4. TRAINING ({args.epochs} epoch(s))")
    _rule("=")
    # The samples are REPLICATED until a few batches are filled. The per-sample
    # training cost depends on the tensor sizes and on the batch, not on what
    # they contain, so replicating them measures the same rate. Without it the
    # phase was almost always left out: at test settings a game yields ~90
    # samples and the real batch is 1024, so twelve games were needed just not to
    # skip it -- and the estimate came out incomplete.
    target = 4 * args.batch
    if 0 < len(samples) < target:
        rounds = -(-target // len(samples))
        print(f"  samples replicated {rounds}x to fill 4 batches of {args.batch}")
        samples = (samples * rounds)[:target]

    m["t_sample"] = 0.0
    if len(samples) < args.batch:
        print(f"  too few samples ({len(samples)}): phase left out of the estimate")
    else:
        net2 = PolicyValueNet(**net_kwargs).to(args.device)
        t0 = time.time()
        train_on_samples(net2, samples, epochs=args.epochs,
                         batch_size=args.batch, device=args.device)
        dt = time.time() - t0
        m["t_sample"] = dt / (len(samples) * args.epochs)
        print(f"  samples                : {len(samples)}")
        print(f"  per sample/epoch       : {m['t_sample']*1000:.3f} ms")
    print()
    return m


def measure_fp16(args, net_kwargs, ckpt: str) -> None:
    """How much half precision pays, measured instead of assumed.

    On recent cards it typically halves the inference cost, but "typically" is
    not a number: it depends on the card, on the batch size and on the network.
    Here the SAME games are played twice, with and without, and the cost per ply
    is compared.

    On the CPU it makes no sense and usually makes things worse: if the device is
    the CPU the test is skipped, saying so.
    """
    import cpp_engine
    from model import PolicyValueNet

    _rule("=")
    print("  HALF PRECISION (fp16)")
    _rule("=")
    if args.device == "cpu":
        print("  skipped: half precision is for the tensor cores of a graphics")
        print("  card. On the CPU it brings nothing and often slows things down.")
        print()
        return

    net = PolicyValueNet(**net_kwargs)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()

    times = {}
    for half in (False, True):
        t0 = time.time()
        try:
            _, out = cpp_engine.selfplay(
                args.engine, net, n_games=args.fp16_games, n_sims=args.sims,
                n_threads=args.threads, device=args.device, fp16=half,
                mcts_batch=args.mcts_batch, max_batch=args.server_batch,
                seed=31337, c_puct=1.5, value_discount=args.discount)
        except Exception as e:
            print(f"  {'fp16' if half else 'fp32'}: failed ({type(e).__name__}: {e})")
            print()
            return
        dt = time.time() - t0
        plies = cpp_engine.parse_stats(out).get("plies_mean") or 0.0
        times[half] = dt / (args.fp16_games * plies) if plies else None
        print(f"  {'fp16' if half else 'fp32'}: {dt:6.1f}s  "
              f"{times[half]*1000:7.2f} ms per ply"
              if times[half] else f"  {'fp16' if half else 'fp32'}: no statistics")

    if times.get(False) and times.get(True):
        r = times[False] / times[True]
        print()
        if r > 1.05:
            print(f"  Half precision is {r:.2f} times faster: worth it.")
            print("    $env:DAMA_CPP_FP16='1'")
        elif r < 0.95:
            print(f"  Half precision makes it SLOWER ({r:.2f} times): leave it off.")
        else:
            print(f"  No appreciable difference ({r:.2f} times).")
        print()
        print("  WARNING: this is a SPEED measurement, not a correctness one.")
        print("  Half precision changes the network's numbers, so before using")
        print("  it in a real run check that play does not get worse:")
        print("  parity_check, or an fp16 against fp32 arena comparison.")
    print()


def measure(args, net_kwargs, ckpt: str) -> dict:
    from parallel import WorkerPool
    from train import train_on_samples

    server = None
    if args.device != "cpu":
        from inference_server import InferenceServer
        server = InferenceServer(net_kwargs, args.device, args.workers,
                                 max_batch=args.server_batch)
        print(f"  inference server on {args.device} "
              f"(max batch {args.server_batch}, {args.workers} processes)\n")

    pool = WorkerPool(net_kwargs, args.workers, server=server)
    torch.set_num_threads(1)   # the main process must not compete for the cores

    m = {}
    try:
        _rule("=")
        print(f"  1. SELF-PLAY ({args.games} games, {args.sims} simulations)")
        _rule("=")
        t0 = time.time()
        games = pool.selfplay(ckpt, args.games, args.sims, args.mcts_batch,
                              12345, c_puct=1.5, value_discount=args.discount)
        dt = time.time() - t0
        good = [g for g in games if g is not None]
        if not good:
            raise SystemExit("  no game completed: configuration to be reviewed")
        plies = [g[2] for g in good]
        m["measured_plies"] = float(np.mean(plies))
        m["t_ply_sp"] = dt / sum(plies)
        print(f"  total time             : {_duration(dt)}")
        print(f"  plies played           : {sum(plies)} "
              f"(mean {m['measured_plies']:.0f} per game)")
        print(f"  cost per ply           : {m['t_ply_sp']*1000:.1f} ms")
        print(f"  cost per game NOW      : {dt/len(good):.1f} s")
        print()

        _rule("=")
        print(f"  2. ARENA ({args.arena} games)")
        _rule("=")
        t0 = time.time()
        pool.arena(ckpt, ckpt, args.arena, args.sims, args.mcts_batch, 999)
        dt = time.time() - t0
        # The arena does not report plies: the same length as self-play is
        # assumed, which is reasonable because they are complete games with the
        # same search on both sides.
        m["t_ply_ar"] = dt / (args.arena * m["measured_plies"])
        print(f"  total time             : {_duration(dt)}")
        print(f"  cost per ply           : {m['t_ply_ar']*1000:.1f} ms")
        print(f"  cost per game NOW      : {dt/args.arena:.1f} s")
        print()

        _rule("=")
        print(f"  3. TRAINING ({args.epochs} epoch(s))")
        _rule("=")
        all_samples = [s for g in good for s in g[0]]
        # Same replication as the compiled path: see the note there.
        target = 4 * args.batch
        if 0 < len(all_samples) < target:
            rounds = -(-target // len(all_samples))
            print(f"  samples replicated {rounds}x to fill 4 batches of {args.batch}")
            all_samples = (all_samples * rounds)[:target]

        m["t_sample"] = 0.0
        if len(all_samples) < args.batch:
            print(f"  too few samples ({len(all_samples)}): phase left out of the estimate")
        else:
            from model import PolicyValueNet
            net = PolicyValueNet(**net_kwargs).to(args.device)
            t0 = time.time()
            train_on_samples(net, all_samples, epochs=args.epochs,
                             batch_size=args.batch, device=args.device)
            dt = time.time() - t0
            m["t_sample"] = dt / (len(all_samples) * args.epochs)
            print(f"  samples                : {len(all_samples)}")
            print(f"  total time             : {_duration(dt)}")
            print(f"  per sample/epoch       : {m['t_sample']*1000:.3f} ms")
        print()
    finally:
        pool.close()
        if server is not None:
            server.close()
    return m


def measure_alphabeta(args, net_kwargs, ckpt: str) -> dict:
    """Fixed-depth opponents, measured separately.

    The conductor ALWAYS plays them on the interpreted, sequential path, even
    when the compiled engine covers self-play and arena: that is why they cost
    disproportionately compared with the rest. Living here, both measurement
    routes can call it.
    """
    m = {}
    # --- fixed-depth opponents ---------------------------------------------
    # SEQUENTIAL path, not the pool: that is how it runs in the run, and it is
    # why it costs disproportionately. It must be measured separately, otherwise
    # it would be estimated with the cost of a normal game and be off by an
    # order of magnitude.
    _rule("=")
    print(f"  4. FIXED-DEPTH OPPONENTS ({args.ab_games} games per level)")
    _rule("=")
    from evaluators import NetEvaluator
    from model import PolicyValueNet
    from players import AlphaBetaPlayer
    from game_diversity import play_one_game

    net = PolicyValueNet(**net_kwargs)
    net.load_state_dict(torch.load(ckpt, map_location=args.device))
    net.eval()
    ev = NetEvaluator(net.to(args.device), args.device)
    m["t_ply_ab"] = {}

    # The DEEP anchor must be measured too, otherwise its cost disappears from
    # the estimate. (A condition that required depth 8 to be among the measured
    # depths never held -- those are 4 and 6 -- and the item silently stayed at
    # zero while the documentation claimed to include it.)
    #
    # It costs several times more per move than the shallower ones, so it is
    # measured with fewer games: a unit cost is needed, not a score.
    levels = [(d, args.ab_games) for d in args.ab_depths]
    if args.deep_every > 0 and args.deep_depth not in args.ab_depths:
        levels.append((args.deep_depth, args.ab_deep_games))

    for depth, how_many in levels:
        opp = AlphaBetaPlayer(depth)
        t0, tot_plies = time.time(), 0
        for g in range(how_many):
            _, _, plies = play_one_game(ev, opp, args.sims, args.mcts_batch,
                                        7000 + g, g % 2 == 0)
            tot_plies += plies
        dt = time.time() - t0
        m["t_ply_ab"][depth] = dt / max(1, tot_plies)
        extra = "  (deep anchor, own cadence)" if depth == args.deep_depth else ""
        print(f"  depth {depth:<2}                 : "
              f"{m['t_ply_ab'][depth]*1000:7.1f} ms per ply "
              f"({dt/max(1,how_many):.1f} s per game now){extra}")
    print()
    return m


def sweep(args, net_kwargs, ckpt: str) -> None:
    """Tries the combinations of the parallelism parameters and says which is
    the fastest ON THIS machine.

    None of these values is tuned for a Windows machine with a graphics card,
    because they were chosen without one to measure on. The defaults come from
    a Linux machine with dozens of processes, where the server's queue is
    always full; with four processes -- the cap imposed on Windows by the
    native crash -- the situation is reversed: the card gets batches of a few
    dozen positions against a cap of 256, and spends most of its time waiting.

    The three levers, and what they do:

      workers    how many play in parallel. On Windows the conductor cannot go
                 beyond 4 without risking the native crash, so trying more here
                 would find an unusable optimum.
      leaves     how many positions each process sends at once. Raising it
                 enlarges the batches without adding processes -- it is the
                 lever left when the processes are stuck at four.
      wait       how long the server waits before starting, in milliseconds.
                 Zero means "go with what there is": with few processes that
                 means tiny batches.
    """
    from parallel import WorkerPool
    _rule("=")
    print("  SEARCH FOR THE FASTEST CONFIGURATION")
    _rule("=")
    print(f"  {args.sweep_games} games per combination, {args.sims} simulations\n")
    print(f"  {'workers':>9}{'leaves':>8}{'wait':>8}{'games/hour':>13}{'':>3}")

    results = []
    for np_ in args.sweep_workers:
        for leaves in args.sweep_leaves:
            for wait in args.sweep_wait:
                server = None
                if args.device != "cpu":
                    from inference_server import InferenceServer
                    server = InferenceServer(net_kwargs, args.device, np_,
                                             max_batch=args.server_batch,
                                             poll_timeout=wait / 1000.0)
                pool = WorkerPool(net_kwargs, np_, server=server)
                try:
                    t0 = time.time()
                    g = pool.selfplay(ckpt, args.sweep_games, args.sims, leaves,
                                      4242, c_puct=1.5, value_discount=args.discount)
                    dt = time.time() - t0
                    good = len([x for x in g if x is not None])
                    ph = 3600 * good / dt if dt > 0 else 0
                    results.append((ph, np_, leaves, wait))
                    print(f"  {np_:>9}{leaves:>8}{wait:>7}ms{ph:>13.0f}")
                finally:
                    pool.close()
                    if server is not None:
                        server.close()

    if not results:
        return
    results.sort(reverse=True)
    ph, np_, leaves, wait = results[0]
    worst = results[-1][0]
    print()
    _rule("=")
    print("  BEST CONFIGURATION")
    _rule("=")
    print(f"  workers={np_}  leaves={leaves}  wait={wait}ms  -> {ph:.0f} games/hour")
    if worst > 0:
        print(f"  ({ph/worst:.2f} times the worst one tried)")
    print()
    print("  To use it in the run:")
    print(f"    $env:DAMA_WORKERS='{np_}'")
    print(f"    $env:DAMA_MCTS_BATCH='{leaves}'")
    print(f"    $env:DAMA_SERVER_WAIT='{wait/1000.0:g}'")
    print()
    print("  Then run the benchmark again without --sweep with these values to")
    print("  get the estimate of the complete run in the chosen configuration.")


def sprt_block(threads: int) -> int:
    """The SPRT block as the conductor computes it.

    Duplicated here on purpose: the benchmark must measure the configuration the
    run will really use, and in the conductor the block is NOT independent of
    the threads -- it is derived from them. Tuning with an arbitrary block would
    measure a configuration that does not exist.
    """
    return max(10, min(100, threads))


def _one_arena(args, cpp_engine, net, work, threads, leaves, n):
    """A single round. Returns (ms per ply, games/hour, batches, plies), or None
    if the engine failed."""
    t0 = time.time()
    try:
        _sc, _wdl, out = cpp_engine.arena(
            args.engine, net, net, n_games=n, n_sims=args.sims,
            n_threads=threads, device=args.device, fp16=args.fp16,
            mcts_batch=leaves, max_batch=args.server_batch, seed=4242,
            temp_plies=args.temp_plies, workdir=work, reuse_jit=True)
    except Exception as e:
        print(f"  {threads:>7}{leaves:>8}{n:>8}   failed: {type(e).__name__}: {e}")
        return None
    dt = time.time() - t0
    plies = cpp_engine.parse_stats(out).get("plies_mean") or 0.0
    return (1000 * dt / (n * plies) if plies else None,
            3600 * n / dt if dt > 0 else 0.0,
            cpp_engine.parse_hub_batches(out), plies)


def _median(v):
    s = sorted(v)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def sweep_arena(args, net_kwargs, ckpt: str) -> None:
    """Tunes the ARENA parameters, the most expensive phase of the cycle.

    Why it deserves its own search and the self-play one is not enough: the arena
    pits TWO different networks against each other, which cannot share a batch.
    So the engine uses two separate hubs, and the card makes two passes instead
    of one, each on a smaller batch. An arena game therefore costs several times
    a self-play game, and the arena ends up the largest item of the cycle.

    THE THREE LEVERS. The THREADS decide how many games are in flight, because
    each thread plays one at a time and stays blocked on its own request. The
    leaf BATCHING decides how large each request is. The BLOCK is not free: the
    conductor derives it from the threads, and that link is reproduced here,
    otherwise a configuration that does not exist would be tuned.

    THE CEILING, worth knowing before turning the knobs: the requests in flight
    are at most as many as the threads and each hub sees about half of them, so
    the maximum batch per hub is (threads / 2) x leaves. The batching actually
    reached stays well below that ceiling.

    WHY EVERY COMBINATION IS REPEATED. Repeating the same command can give
    costs per ply further apart than the configurations being compared, which
    differ by little. A single round does not measure, it probes -- and whoever
    tunes on a remote machine does one round and trusts it. So it is repeated,
    the median is reported with the spread, and when the first two cannot be
    told apart it is SAID instead of declaring a winner.
    """
    import cpp_engine
    from model import PolicyValueNet

    if not args.engine:
        raise SystemExit(
            "  the arena search needs the compiled engine (--engine):\n"
            "  it is the only path in which the arena has two separate hubs,\n"
            "  i.e. the phenomenon to tune.")

    net = PolicyValueNet(**net_kwargs)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()

    _rule("=")
    print("  ARENA TUNING (the most expensive phase of the cycle)")
    _rule("=")
    print(f"  {args.sims} simulations per move, network {args.channels}x{args.blocks}")
    print(f"  {args.arena_rounds} rounds per combination, to tell a gain "
          f"from the noise")
    if args.arena_block == [0]:
        print("  block: the one the conductor would derive from the threads")
    else:
        print(f"  forced block: {args.arena_block}")
    print()

    # The model is exported ONCE, before the clock starts: otherwise every round
    # would pay again, inside its own time, a fixed cost equal for all, which
    # dilutes exactly the difference to be measured. The two copies are the same
    # network -- here the batching mechanics are tuned, not who wins.
    work = tempfile.mkdtemp(prefix="dama_arena_")
    planes = net_kwargs.get("in_planes", 7)
    cpp_engine.export_jit(net, os.path.join(work, "arena_a.pt"), planes)
    cpp_engine.export_jit(net, os.path.join(work, "arena_b.pt"), planes)

    print(f"  {'threads':>7}{'leaves':>8}{'block':>8}{'batch':>8}{'plies':>8}"
          f"{'ms/ply':>11}{'spread':>10}{'games/hour':>13}")

    results = []
    try:
        for threads in args.arena_threads:
            for leaves in args.arena_leaves:
                for block in args.arena_block:
                    n = sprt_block(threads) if block == 0 else block
                    rounds = [_one_arena(args, cpp_engine, net, work,
                                         threads, leaves, n)
                              for _ in range(args.arena_rounds)]
                    rounds = [g for g in rounds if g and g[0]]
                    if not rounds:
                        continue
                    msl = [g[0] for g in rounds]
                    ms, lo, hi = _median(msl), min(msl), max(msl)
                    ph = _median([g[1] for g in rounds])
                    batches = [x for g in rounds for x in g[2]]
                    bat = _median(batches) if batches else 0.0
                    plies = _median([g[3] for g in rounds])
                    # The spread is the only thing that says whether the number
                    # next to it means anything: without it, two random
                    # measurements look like a ranking.
                    spread = (hi - lo) / ms if ms else 0.0
                    results.append((ms, lo, hi, ph, threads, leaves, n, bat, plies))
                    print(f"  {threads:>7}{leaves:>8}{n:>8}{bat:>8.1f}{plies:>8.0f}"
                          f"{ms:>11.2f}{100*spread:>9.0f}%{ph:>13.0f}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if not results:
        return
    results.sort(key=lambda r: r[0])
    ms, lo, hi, ph, threads, leaves, n, bat, plies = results[0]

    print()
    _rule("=")
    print("  BEST CONFIGURATION FOR THE ARENA")
    _rule("=")

    # Two combinations can be told apart only if their intervals do not overlap.
    # With few rounds and a busy machine this is often not the case, and
    # declaring a winner then means making one up.
    uncertain = len(results) > 1 and results[1][1] <= hi
    if uncertain:
        print("  THE MEASUREMENT CANNOT TELL THE FIRST TWO APART.")
        print(f"    threads={threads} leaves={leaves}: {lo:.1f}-{hi:.1f} ms")
        s = results[1]
        print(f"    threads={s[4]} leaves={s[5]}: {s[1]:.1f}-{s[2]:.1f} ms")
        print("  The intervals overlap: the difference between the two lies")
        print("  within the noise. Raise --arena-rounds or --arena-block, or")
        print("  take the simpler of the two -- there is no evidence that the")
        print("  other one is better.")
        print()

    print(f"  threads={threads}  leaves={leaves}  block={n}  ->  {ms:.2f} ms per "
          f"ply ({lo:.1f}-{hi:.1f}), batch {bat:.1f}")
    worst = results[-1][0]
    if ms > 0:
        print(f"  ({worst/ms:.2f} times the worst one tried)")
    print()
    print("  To use it in the run:")
    print(f"    $env:DAMA_CPP_THREADS='{threads}'")
    print(f"    $env:DAMA_MCTS_BATCH='{leaves}'")
    if n != sprt_block(threads):
        print(f"    $env:DAMA_SPRT_CHUNK='{n}'   # different from the one derived from the threads")
    print()
    print("  WARNING: DAMA_MCTS_BATCH applies to ALL phases, not just the")
    print("  arena. If the best value here makes self-play worse, look at")
    print("  which of the two weighs more in the cycle breakdown: on a real")
    print("  run the arena was 65% and self-play 29%.")
    print()
    print("  WARNING, the most important one: batching is NOT free. More")
    print("  leaves per batch mean fewer times the tree reacts to its own")
    print("  results, and the search gets worse. MEASURED, the same trained")
    print("  network on both sides, 400 simulations, 48 games per comparison:")
    print("     8 leaves against 32 -> 0.802 for the 8   (32 leaves: -243 Elo)")
    print("     8 leaves against 64 -> 0.844 for the 8   (64 leaves: -293 Elo)")
    print("  62 wins, 34 draws, ZERO losses over 96 games.")
    print()
    print("  The reason is draughts: with 5.84 legal moves on average, a batch")
    print("  of 64 leaves forces the search to spread over the whole tree.")
    print("  In chess, with a branching factor of 35, it distorts much less.")
    print()
    print("  So the speed above is real but it is paid in playing strength,")
    print("  and paid for the whole run. Before raising the leaves, measure")
    print("  what they cost: the engine has --mcts-batch-b for that.")

    # An optimum falling on the EDGE of the grid is not an optimum: it is the
    # point where the grid ended. Saying so is the only difference between a
    # measurement and a number that looks like one.
    edges = []
    if len(args.arena_leaves) > 1 and leaves in (min(args.arena_leaves),
                                                 max(args.arena_leaves)):
        direction = "lower" if leaves == min(args.arena_leaves) else "higher"
        edges.append(f"--arena-leaves: the best is {leaves}, which is the"
                     f" edge. Try again with {direction} values.")
    if len(args.arena_threads) > 1 and threads in (min(args.arena_threads),
                                                   max(args.arena_threads)):
        direction = "lower" if threads == min(args.arena_threads) else "higher"
        edges.append(f"--arena-threads: the best is {threads}, which is the"
                     f" edge. Try again with {direction} values.")
    if edges and not uncertain:
        print()
        print("  THE MAXIMUM IS NOT BRACKETED BY THE GRID")
        for b in edges:
            print(f"    {b}")
        print("  As long as the optimum sits on the edge, the grid says on which")
        print("  side the maximum is, not where.")

def estimate(args, m: dict) -> float:
    _rule("=")
    print("  ESTIMATE OF THE COMPLETE RUN")
    _rule("=")

    # With an already trained network the measured length IS the steady-state
    # one, and using it instead of a table value is the whole reason why
    # --weights is worth passing. With a random network instead the measurement
    # does not represent the run and is replaced by the known steady-state length.
    measured = m["measured_plies"]
    # NEVER ASSUME THAT GAMES WILL GET SHORTER. The correction exists because a
    # weak network plays short games and would underestimate the steady-state
    # cost. But the length does not depend only on the network's strength: an
    # untrained network can also produce games LONGER than the steady-state ones,
    # and applying the correction blindly would then cut the estimate for no
    # reason -- an invented gain, and in the dangerous direction to boot.
    #
    # With a trained network the measured length is used, which is the good
    # figure. Without one, the LONGER of the measured and the steady-state length
    # is taken.
    L = measured if args.weights else max(measured, args.plies)
    print(f"  length measured here     : {measured:.0f} plies")
    # Length is the noisiest quantity of the whole measurement: between two
    # rounds of a few games 54 and 216 plies were seen. With a small sample it
    # must be said, otherwise a mean of three games is taken as precise.
    if args.games < 6:
        print(f"  WARNING: mean over only {args.games} games, very noisy.")
        print("  Length is the quantity that weighs most on the estimate: with")
        print("  --games 12 or more the number becomes reliable.")
    if args.weights:
        print(f"  length used              : {L:.0f} plies (the measured one)")
        print("  The given network is trained, so its games already have the")
        print("  steady-state length: they do not need correcting.")
        if args.sims < 200:
            print(f"  WARNING: with only {args.sims} simulations play is poor and")
            print("  games end early. Run again with the real values (--sims 400)")
            print("  or the estimate will come out too low.")
    else:
        print(f"  length used              : {L:.0f} steady-state plies "
              f"(factor {L/measured:.2f})")
        print("  Games get longer as the network learns: with a random network")
        print("  ~84 plies are measured, with a trained one 150-159.")
        print("  With --weights on a trained champion this correction is not needed.")
    print()

    selfplay = args.games_per_cycle * L * m["t_ply_sp"]
    arena = args.arena_games_per_cycle * L * m["t_ply_ar"]
    training = args.window * args.epochs * m["t_sample"]

    # Generation ladder: arena games against the last anchors, every GEN_EVERY
    # cycles. Easy to forget, and not small.
    ladder = (args.gen_games * args.gen_anchors * L * m["t_ply_ar"]
              / max(1, args.gen_every))

    # Strength test: two levels every STRENGTH_EVERY cycles, plus the deep
    # anchor on its own cadence.
    strength = 0.0
    for depth in args.ab_depths:
        strength += args.strength_games * L * m["t_ply_ab"][depth] / max(1, args.strength_every)
    if args.deep_depth in m["t_ply_ab"] and args.deep_every > 0:
        strength += (args.deep_games * L * m["t_ply_ab"][args.deep_depth]
                     / args.deep_every)

    # Exporting the model to TorchScript: the conductor redoes it every cycle to
    # hand the network to the engine. Small, but it belongs to the cycle.
    export = m.get("t_export", 0.0)
    cycle = selfplay + arena + training + ladder + strength + export
    items = (("self-play", selfplay), ("arena", arena),
             ("training", training), ("generation ladder", ladder),
             ("strength test", strength), ("model export", export))
    print(f"  {'phase':<26}{'per cycle':>14}{'share':>9}")
    for name, v in items:
        print(f"  {name:<26}{_duration(v):>14}{100*v/cycle:>8.0f}%")
    _rule()
    print(f"  {'MEAN CYCLE':<26}{_duration(cycle):>14}")
    print()

    total = args.cycles * cycle
    print(f"  {'cycles':>8}{'estimated duration':>22}")
    for n in sorted({10, 50, args.cycles, 200}):
        print(f"  {n:>8}{_duration(n * cycle):>22}")
    print()

    # The dominant uncertainty is not the clock but the length of the games:
    # stating it is more honest, and more useful, than a single figure.
    if args.weights:
        low, high = total * 0.90, total * 1.15
        explain = ("  A narrow interval because the length of the games was measured\n"
                   "  and not assumed. What remains is the margin of the small sample\n"
                   "  and of the residual growth in the final cycles.")
    else:
        # min() and not the plain ratio: if the measured games are already longer
        # than the steady-state ones the ratio exceeds 1 and the "low" end would
        # come out ABOVE the high one -- a reversed interval.
        low = total * min(1.0, measured / L)
        high = total * 1.15
        explain = ("  The low end holds if games stayed as long as those just\n"
                   "  measured -- i.e. only for the very first cycles. The dominant\n"
                   "  uncertainty is the length of the games, not the clock: with\n"
                   "  --weights on a trained champion it narrows a lot.")
    low, high = min(low, high), max(low, high)
    _rule("=")
    print(f"  COMPLETE RUN ({args.cycles} cycles): {_duration(total)}"
          + ("   *** INCOMPLETE ESTIMATE ***" if m.get("t_sample", 0) <= 0 else ""))
    _rule("=")
    print(f"  plausible interval: from {_duration(low)} to {_duration(high)}")

    # A phase left out of the estimate cannot go unnoticed in the total. Training
    # is skipped when the test games produce fewer samples than a batch -- it
    # happens with few games and few simulations -- and without a warning the
    # total looks complete. Training can be worth a large share of the cycle, so
    # an estimate that silently leaves it out is not comparable with one that
    # includes it.
    if m.get("t_sample", 0) <= 0:
        print()
        print("  WARNING: TRAINING is not included in these numbers.")
        print("  The test games produced fewer samples than a batch, so the")
        print("  phase was skipped. It can be a very large share of the cycle,")
        print("  so the estimate is TOO LOW. Run again with more games")
        print("  (--games 8) or with a smaller batch (--batch 64).")
    print()
    print(explain)
    return cycle


def main():
    ap = argparse.ArgumentParser(
        description="Estimates how long the complete run would take on this machine.")
    ap.add_argument("--device", default=os.environ.get("DAMA_DEVICE", "cpu"))
    ap.add_argument("--weights", default="", help="trained champion: makes the estimate more precise")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--arena", type=int, default=6)
    ap.add_argument("--ab-games", type=int, default=2)
    ap.add_argument("--ab-depths", type=int, nargs="+", default=[4, 6])
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--workers", type=int, default=0, help="0 = as the conductor")
    ap.add_argument("--mcts-batch", type=int, default=8)
    ap.add_argument("--server-batch", type=int, default=256)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=2)
    # MUST match the conductor's batch (DAMA_BATCH, default 1024): the per-sample
    # cost depends heavily on the batch -- 28 ms with 64, 5.5 with 256 on the
    # same machine -- and it is multiplied by over a million.
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--discount", type=float, default=0.99)
    # compiled path: it is the production one when the engine is there
    ap.add_argument("--engine", default=os.environ.get("DAMA_CPP_ENGINE", ""),
                    help="dama_engine executable: if given, THIS path is measured")
    ap.add_argument("--threads", type=int, default=0, help="engine threads (0 = all the cores)")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--temp-plies", type=int, default=10)
    ap.add_argument("--cores", action="store_true", default=True,
                    help="measure the usable cores (default)")
    ap.add_argument("--no-cores", dest="cores", action="store_false")
    ap.add_argument("--fp16-compare", action="store_true",
                    help="compare half and full precision")
    ap.add_argument("--fp16-games", type=int, default=4)
    # the real run the estimate projects onto (defaults = the conductor's)
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--games-per-cycle", type=int, default=400)
    ap.add_argument("--arena-games-per-cycle", type=int, default=226,
                    help="mean measured on a real run: the SPRT stops before the cap")
    ap.add_argument("--window", type=int, default=600000)
    ap.add_argument("--plies", type=float, default=STEADY_PLIES)
    ap.add_argument("--gen-every", type=int, default=10)
    ap.add_argument("--gen-games", type=int, default=100)
    ap.add_argument("--gen-anchors", type=int, default=3)
    ap.add_argument("--strength-every", type=int, default=20)
    ap.add_argument("--strength-games", type=int, default=60)
    ap.add_argument("--deep-depth", type=int, default=8)
    ap.add_argument("--deep-every", type=int, default=60)
    ap.add_argument("--deep-games", type=int, default=20)
    ap.add_argument("--ab-deep-games", type=int, default=1,
                    help="test games for the deep anchor: it costs much more")
    # search for the best configuration
    ap.add_argument("--sweep", action="store_true",
                    help="try the parallelism combinations and report the fastest")
    ap.add_argument("--sweep-workers", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--sweep-leaves", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--sweep-wait", type=float, nargs="+", default=[0, 2, 5],
                    help="server wait in milliseconds")
    ap.add_argument("--sweep-games", type=int, default=4)
    # ARENA tuning: levers, values and block size
    ap.add_argument("--sweep-arena", action="store_true",
                    help="tune the arena parameters, the most expensive phase")
    # Measures only the cores and exits. It exists because the preflight had ITS
    # OWN route for the same measurement, which called usable_cores.py with no
    # deadline and without closing the child processes: two implementations, only
    # one of them protected. Better a single one, the one already verified.
    ap.add_argument("--cores-only", action="store_true",
                    help="measure the usable cores and stop")
    # By default it SAVES. Whoever measures on a remote machine has no way of
    # knowing which lines the reader will need, and an extra report on disk never
    # did any harm; a lost measurement does.
    ap.add_argument("--report", default="auto",
                    help="file to save everything to ('no' to not save)")
    ap.add_argument("--seed", type=int, default=20260821,
                    help="seed of the fallback network: two launches with the same"
                         " seed are comparable")
    ap.add_argument("--repeats", type=int, default=3,
                    help="rounds of the core measurement")
    ap.add_argument("--arena-threads", type=int, nargs="+", default=[0],
                    help="0 = use --threads")
    # Up to 64 because at 32 the optimum fell on the EDGE of the grid, and a grid
    # that does not bracket the maximum has not measured it: it only says on
    # which side it is. On a card, where large batches pay more than on the CPU,
    # the maximum can be even further out.
    ap.add_argument("--arena-leaves", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--arena-block", type=int, nargs="+", default=[0],
                    help="games per SPRT block; 0 = the one derived from the threads")
    ap.add_argument("--arena-rounds", type=int, default=3,
                    help="repetitions per combination: below three a gain cannot"
                         " be told from the noise")
    args = ap.parse_args()

    # Opened BEFORE any print: a report that starts halfway does not contain the
    # environment, which is the first thing to look at when the numbers come
    # from a machine that is not at hand.
    if args.report != "no":
        open_report(args.report)

    if args.workers <= 0:
        args.workers = min(max(1, (os.cpu_count() or 2) - 1), 64)
        # On Windows a large pool can make the process die of a native crash
        # during training (docs/running.md): the benchmark must measure a configuration
        # the run can really use.
        # The cap applies only to the PYTHON pool: with the compiled engine that
        # pool is not even created, and the native crash cannot happen.
        if os.name == "nt" and args.workers > 4 and not args.engine:
            print(f"  [Windows] workers reduced from {args.workers} to 4: with more than")
            print("            four the run can die of a native crash (docs/running.md).\n")
            args.workers = 4

    environment(args.device)

    # The cores are MEASURED. On rented machines the declared and the usable
    # ones do not coincide, and sizing the engine threads on the declared number
    # means asking for more than there are.
    usable = None
    if args.cores_only:
        args.cores = True       # it is the only thing that was asked
    if args.cores:
        _rule('=')
        print("  USABLE CORES")
        _rule('=')
        # usable_cores.py already prints everything (declared, quota, spread,
        # suggestion): nothing is repeated here, only the value is read.
        usable = measure_usable_cores(args.repeats)
        if usable is None:
            print("  without this measurement the declared cores are used, which on a")
            print("  shared machine can be many more than the usable ones.")
        print()

    if args.cores_only:
        raise SystemExit(0 if usable else 1)

    if args.engine and args.threads <= 0:
        args.threads = max(1, int(usable)) if usable else (os.cpu_count() or 4)
        print(f"  engine threads not given: using {args.threads}")
        print()

    # FEWER GAMES THAN THREADS = the sample is measured, not the machine.
    #
    # In the engine each thread takes one game at a time: with 8 games and 12
    # threads, four threads stay idle the whole time and the requests in flight --
    # those that fill the batches sent to the card -- are eight instead of
    # twelve. The resulting cost per ply is that of a smaller machine, so the
    # benchmark would underestimate the machine and overestimate the duration of
    # a run that plays hundreds of games per cycle.
    if args.engine and args.threads > 0:
        for name in ("games", "arena", "fp16_games"):
            if getattr(args, name) < args.threads:
                old = getattr(args, name)
                setattr(args, name, args.threads)
                print(f"  {name.replace('_', ' ')}: {old} -> {args.threads} "
                      f"(fewer than the threads: {args.threads - old} would have"
                      f" stayed idle)")
        print()

    net_kwargs = dict(channels=args.channels, n_blocks=args.blocks)

    from model import PolicyValueNet
    work = tempfile.mkdtemp(prefix="dama_bench_")
    ckpt = os.path.join(work, "net.pt")
    if args.weights:
        # The file must be CHECKED here, not found wrong downstream. If it is not
        # a valid network the worker processes fail one by one and the benchmark
        # only says "no game completed: configuration to be reviewed" -- true but
        # useless, because it sends one looking for the problem everywhere except
        # in the file just given.
        if not os.path.exists(args.weights):
            raise SystemExit(f"  weights file does not exist: {args.weights}")
        try:
            sd = torch.load(args.weights, map_location="cpu")
            PolicyValueNet(**net_kwargs).load_state_dict(sd)
        except Exception as e:
            raise SystemExit(
                f"  {args.weights} is not a valid {args.channels}x{args.blocks} network:\n"
                f"    {type(e).__name__}: {str(e).splitlines()[0]}\n"
                f"  A champion saved by the run is needed (for example champion.pt or\n"
                f"  gen_NNN.pt). If the architecture differs, give it with\n"
                f"  --channels and --blocks.")
        shutil.copyfile(args.weights, ckpt)
        print(f"  starting network: {args.weights}\n")
    else:
        # The fallback network is seeded. Without it, every launch of the
        # benchmark created a different one: the games changed, their LENGTH
        # changed and with it the cost per ply -- shorter games end with more
        # pieces on the board, hence more moves to evaluate at every node. Two
        # runs of the same command could then reach different conclusions on
        # which batching pays. A benchmark that cannot be repeated does not
        # measure, it probes.
        torch.manual_seed(args.seed)
        torch.save(PolicyValueNet(**net_kwargs).state_dict(), ckpt)
    try:
        t0 = time.time()
        if args.sweep_arena:
            if args.arena_threads == [0]:
                args.arena_threads = [args.threads if args.threads > 0 else 4]
            sweep_arena(args, net_kwargs, ckpt)
            print(f"\n  (tuning completed in {_duration(time.time() - t0)})")
            return
        if args.sweep:
            sweep(args, net_kwargs, ckpt)
            print(f"\n  (search completed in {_duration(time.time() - t0)})")
            return
        if args.engine:
            if not os.path.exists(args.engine):
                raise SystemExit(f"  engine not found: {args.engine}")
            print(f"  compiled engine: {args.engine}")
            print("  (the PRODUCTION path is measured, not the interpreted one)")
            print()
            if args.fp16_compare:
                measure_fp16(args, net_kwargs, ckpt)
            m = measure_engine(args, net_kwargs, ckpt)
            m.update(measure_alphabeta(args, net_kwargs, ckpt))
        else:
            m = measure(args, net_kwargs, ckpt)
            m.update(measure_alphabeta(args, net_kwargs, ckpt))
        cycle = estimate(args, m)
        print()
        # A single copyable line: without it, a screenful to interpret comes back.
        _rule("=")
        print("  LINE TO SEND BACK TO WHOEVER ASKED FOR THE MEASUREMENT")
        _rule("=")
        # With the compiled engine the number that matters is the THREADS, not the
        # processes of the Python pool -- which in that case does not exist.
        # Reporting the wrong value in a line made to be copied is worse than
        # saying nothing.
        how_many = (f"threads={args.threads}" if args.engine else f"workers={args.workers}")
        incomplete = " | NO training" if m.get("t_sample", 0) <= 0 else ""
        print(f"  device={args.device} {how_many} sims={args.sims} "
              f"net={args.channels}x{args.blocks} | cycle={_duration(cycle)} | "
              f"{args.cycles} cycles = {_duration(args.cycles * cycle)}{incomplete}")
        print()
        print(f"  (benchmark completed in {_duration(time.time() - t0)})")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
