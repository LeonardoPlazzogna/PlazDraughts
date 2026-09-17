<#
preflight.ps1 - THE SINGLE COMMAND to give before a run.
Windows mirror of preflight.sh.

In seven phases it checks everything there is to check and estimates how long
the run would take:

  0. environment: torch, CUDA, consistency with the requested device
  1. build of the C++ engine and its validation
  2. all the Python tests (discovered from the folder, not listed by hand)
  3. cores REALLY available, not the declared ones
  4. benchmark: estimated duration of the complete run
  5. resume after an interruption
  6. state of the LibTorch backend

It ends with a single line to send back to whoever asked for the measurement.

Phase 4 delegates to benchmark.py instead of measuring on its own: the previous
version ran two whole conductor cycles -- an hour -- and measured the
interpreted path even when the compiled engine was available, that is, the
wrong configuration.

USAGE:
  .\preflight.ps1
  $env:DAMA_CPP_ENGINE = 'engine_c\build\dama_engine.exe'   # measures the real engine
  $env:DAMA_DEVICE = 'cuda'
  $env:DAMA_WEIGHTS_FILE = 'weights\champion.pt'            # more precise estimate
  .\preflight.ps1
#>
param(
    [int]$BenchGames = $(if ($env:BENCH_GAMES) { [int]$env:BENCH_GAMES } else { 8 }),
    [int]$BenchSims = $(if ($env:BENCH_SIMS) { [int]$env:BENCH_SIMS } else { 400 }),
    # MUST match the real value of the arena games (30): the arena costs as much
    # as self-play, so measuring it with fewer games removes most of that phase
    # from the fixed cost of EVERY cycle and the estimate comes out far too low.
    [int]$BenchArenaGames = $(if ($env:BENCH_ARENA_GAMES) { [int]$env:BENCH_ARENA_GAMES } else { 30 }),
    # Reduced pool by default: with the default number of workers (cpu_count-1)
    # and a full-size network, training can crash natively on Windows -- see
    # "The known crash on Windows" in docs/running.md. 4 is the value verified
    # as stable. Raise it if your machine handles the default.
    [int]$BenchWorkers = $(if ($env:BENCH_WORKERS) { [int]$env:BENCH_WORKERS } else { 4 }),
    # Trained champion: it makes the estimate much more precise, because the
    # length of the games is MEASURED instead of assumed.
    [string]$Weights = $env:DAMA_WEIGHTS_FILE,
    # Arena tuning: NOT on by default, because it tries one combination after
    # another and takes a good ten minutes, while the rest of the preflight is a
    # sanity check. It is done once only on the machine that will run, before
    # launching the real run: the arena is two thirds of the cycle, so that is
    # where a tuning pays for itself.
    [switch]$TuneArena
)

$root = $PSScriptRoot
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$report = Join-Path $root "preflight_report_$ts.txt"
# Every write to the report declares UTF8. Out-File without -Encoding writes
# UTF-16 in Windows PowerShell 5.1, and the build, test and benchmark sections
# appended that way came out unreadable in the middle of a UTF-8 file -- the
# very report meant for whoever reads the results from elsewhere.
"" | Out-File -FilePath $report -Encoding UTF8

$script:pass = 0
$script:fail = 0
function Log($m) { Write-Host $m; $m | Out-File -FilePath $report -Append -Encoding UTF8 }
function Hr($m)  { Log ""; Log ("=" * 60); Log $m; Log ("=" * 60) }
function Ok($m)  { Log "  [PASS] $m"; $script:pass++ }
function Ko($m)  { Log "  [FAIL] $m"; $script:fail++ }

Hr "PREFLIGHT - $ts"
Log "root=$root"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) { Log "ERROR: python is not on the PATH."; exit 1 }

# A C++20 compiler is needed, not a particular one. This line used to demand
# g++ and exit: on a Windows machine with Visual Studio -- that is, the very one
# that will do the run -- the preflight died on its first line without having
# tried anything, saying that a compiler is missing that is not needed there.
$hasGpp = [bool](Get-Command g++ -ErrorAction SilentlyContinue)
$hasCl  = [bool](Get-Command cl -ErrorAction SilentlyContinue)
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not $hasCl -and (Test-Path $vswhere)) {
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if ($vs) { $hasCl = $true }
}
$libtorch = if ($env:LIBTORCH) { $env:LIBTORCH } else { $env:DAMA_LIBTORCH }
Log "  compilers: g++ $(if ($hasGpp) {'yes'} else {'no'})   MSVC $(if ($hasCl) {'yes'} else {'no'})"
Log "  LIBTORCH : $(if ($libtorch) { $libtorch } else { 'not set' })"
if (-not $hasGpp -and -not $hasCl) {
    Log "ERROR: no C++20 compiler found (neither g++ nor Visual Studio)."
    exit 1
}
# build.ps1 chooses by itself: with LIBTORCH set it takes MSVC and the real
# backend, otherwise it falls back to g++ and the fake backend. If there is only
# MSVC but LIBTORCH is not set, that fallback does not exist and it must be said
# now.
if (-not $libtorch -and -not $hasGpp) {
    Log "ERROR: MSVC is there but LIBTORCH is not set, and without g++ there is no"
    Log "  possible fallback. Download LibTorch for Windows and set:"
    Log "    `$env:LIBTORCH = 'C:\libtorch'"
    exit 1
}
if (-not $libtorch) {
    Log "  NOTE: without LIBTORCH the engine is built with the FAKE backend."
    Log "  Good for validating the rules, NOT for estimating the times of a run."
}

# ============================================================
# 0. ENVIRONMENT (python, torch, CUDA)
# ============================================================
Hr "0. ENVIRONMENT (python, torch, CUDA)"
$envScript = @'
import os, sys
print(f"  python : {sys.version.split()[0]}")
try:
    import torch
except Exception as e:
    print(f"  torch  : CANNOT BE IMPORTED ({e})")
    sys.exit(2)
print(f"  torch  : {torch.__version__}")
avail = torch.cuda.is_available()
print(f"  CUDA available : {avail}")
if avail:
    print(f"  CUDA runtime   : {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}          : {torch.cuda.get_device_name(i)}")
dev = os.environ.get("DAMA_DEVICE", "cpu")
srv = os.environ.get("DAMA_INFERENCE_SERVER", "auto")
print(f"  DAMA_DEVICE = {dev}   DAMA_INFERENCE_SERVER = {srv}")
if dev != "cpu" and not avail:
    print(f"  ERROR: DAMA_DEVICE={dev} but torch does NOT see any GPU.")
    print("  Install the right CUDA build of PyTorch (see requirements.txt).")
    sys.exit(1)
if dev != "cpu":
    print("  -> GPU path: the inference server will be ACTIVE (inference_server.py)")
else:
    print("  -> CPU path: pool of workers with one copy of the network each")
'@
$envPath = Join-Path $env:TEMP "preflight_env_check.py"
$envScript | Set-Content -Encoding utf8 $envPath
$envOut = & python $envPath 2>&1 | Out-String
$envRc = $LASTEXITCODE
Log $envOut.TrimEnd()
Remove-Item -Force -ErrorAction SilentlyContinue $envPath
if ($envRc -eq 0) { Ok "environment consistent with DAMA_DEVICE" } else { Ko "environment NOT consistent (see above)" }

# ============================================================
# 1. BUILD + VALIDATION of engine_c/ (C++)
# ============================================================
Hr "1. BUILD + VALIDATION of engine_c/ (C++ engine: rules, encoder, MCTS, concurrency, self-play)"
$engc = Join-Path $root "engine_c"
Push-Location $engc
$buildOut = & .\build.ps1 2>&1 | Out-String
$buildRc = $LASTEXITCODE
Pop-Location
$buildOut | Out-File -Append $report -Encoding UTF8
if ($buildRc -eq 0 -and $buildOut -notmatch "WRONG|MISMATCH|Segmentation") {
    Ok "engine_c: perft, encoder, MCTS, concurrency, multi-thread self-play, arena"
} else {
    Ko "engine_c build/validation did NOT pass (rc=$buildRc, see report)"
    ($buildOut -split "`n" | Select-Object -Last 15) | ForEach-Object { Log "    $_" }
}
Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $engc "perft.exe"), (Join-Path $engc "encoder_parity.exe"), (Join-Path $engc "mcts_selfplay_test.exe")

# ============================================================
# 2. PYTHON TESTS
# ============================================================
Hr "2. PYTHON TESTS (rules, AlphaZero end-to-end, inference server)"
# The tests are DISCOVERED, not listed. Listed by hand they were eight out of
# the ten on disk: whoever adds one does not see it run, and the gate that
# declares "all green" does not check everything. Found by counting the files.
$test = @(Get-ChildItem (Join-Path $root "tests") -Filter "test_*.py" |
          Sort-Object Name | ForEach-Object { $_.FullName })
Log "  ($($test.Count) test files found)"
foreach ($t in $test) {
    $out = & python $t 2>&1 | Out-String
    $rc = $LASTEXITCODE
    $out | Out-File -Append $report -Encoding UTF8
    if ($rc -eq 0 -and $out -notmatch "FAIL") {
        Ok "$(Split-Path $t -Leaf)"
    } else {
        Ko "$(Split-Path $t -Leaf) (rc=$rc)"
        ($out -split "`n" | Select-Object -Last 8) | ForEach-Object { Log "    $_" }
    }
}

# ============================================================
# 3. CORES REALLY AVAILABLE
# ============================================================
Hr "3. CORES REALLY AVAILABLE"
# It is not a detail: on a shared machine the declared cores and the usable ones
# do not coincide, and the engine threads must be sized on the latter. It goes
# through benchmark.py and not through usable_cores.py directly: the measurement
# saturates the machine with dozens of processes and may not finish -- a pool
# was observed hanging for more than twenty hours. benchmark.py runs it with a
# deadline and, if the deadline passes, closes the WHOLE process tree. Calling
# usable_cores.py from here bypassed that protection, and a stuck preflight
# prints nothing: whoever launches it remotely only sees a frozen command.
$coresOut = & python (Join-Path $root "benchmark.py") --cores-only --repeats 3 2>&1 | Out-String
$coresOut | Out-File -Append $report -Encoding UTF8
Log $coresOut.TrimEnd()
$threads = 0
if ($coresOut -match "DAMA_CPP_THREADS=(\d+)") { $threads = [int]$Matches[1] }
if ($threads -gt 0) { Ok "cores measured ($threads threads suggested)" }
else { Ko "core measurement did not succeed" }

# ============================================================
# 4. BENCHMARK: how long the complete run would take
# ============================================================
Hr "4. BENCHMARK (estimated duration of the complete run)"
# It delegates to benchmark.py instead of redoing a measurement of its own here:
# a second implementation drifts from the first, and the one that lived here ran
# whole conductor cycles and measured the interpreted path even when the compiled
# engine was available, that is, the wrong configuration.
# The arena is measured with ITS games, not with the self-play ones:
# $BenchArenaGames carries the real value and must reach benchmark.py.
$benchArgs = @("--games", "$BenchGames", "--arena", "$BenchArenaGames",
               "--sims", "$BenchSims", "--no-cores")
if ($env:DAMA_CPP_ENGINE -and (Test-Path $env:DAMA_CPP_ENGINE)) {
    $benchArgs += @("--engine", $env:DAMA_CPP_ENGINE)
    if ($threads -gt 0) { $benchArgs += @("--threads", "$threads") }
    if ($env:DAMA_DEVICE -and $env:DAMA_DEVICE -ne "cpu") { $benchArgs += "--fp16-compare" }
} else {
    $benchArgs += @("--workers", "$BenchWorkers")
}
if ($env:DAMA_DEVICE) { $benchArgs += @("--device", $env:DAMA_DEVICE) }
if ($Weights) { $benchArgs += @("--weights", $Weights) }
$benchOut = & python (Join-Path $root "benchmark.py") @benchArgs 2>&1 | Out-String
$benchRc = $LASTEXITCODE
$benchOut | Out-File -Append $report -Encoding UTF8
Log $benchOut.TrimEnd()
if ($benchRc -eq 0) { Ok "benchmark completed" } else { Ko "benchmark did NOT complete (rc=$benchRc)" }
$summary = ($benchOut -split "`n" | Where-Object { $_ -match "100 cycles =" } | Select-Object -Last 1)

# ============================================================
# 4b. ARENA TUNING (on request only)
# ============================================================
# Why it is separate from the benchmark of point 4: that one MEASURES the
# current configuration, this one LOOKS FOR a better one by trying several.
# They are two different questions and they cost differently.
$tuningDone = $false
if ($TuneArena) {
    Hr "4b. ARENA TUNING (most expensive phase of the cycle)"
    if (-not ($env:DAMA_CPP_ENGINE -and (Test-Path $env:DAMA_CPP_ENGINE))) {
        Log "  skipped: it needs the compiled engine (DAMA_CPP_ENGINE)."
        Log "  It is the only path in which the arena keeps two separate hubs, that"
        Log "  is, the phenomenon to tune."
    } else {
        $tBase = if ($threads -gt 0) { $threads } else { [Environment]::ProcessorCount }
        $tList = @("$tBase")
        if ($tBase * 2 -le 128) { $tList += "$($tBase * 2)" }
        # The simulations stay the REAL ones of the run: changing them changes
        # the ratio between search work and network work, that is, precisely the
        # balance to tune. The number of combinations is shortened, not the
        # fidelity of each one.
        $tuneArgs = @("--engine", $env:DAMA_CPP_ENGINE, "--sweep-arena",
                      "--arena-threads") + $tList + @(
                      "--arena-leaves", "8", "16", "32", "64",
                      "--sims", "$BenchSims")
        Log ("  " + ($tList.Count * 4) + " combinations at $BenchSims simulations:")
        Log "  it can take quite a few minutes, and that is normal. It is done once only."
        if ($env:DAMA_DEVICE) { $tuneArgs += @("--device", $env:DAMA_DEVICE) }
        if ($Weights) { $tuneArgs += @("--weights", $Weights) }
        $tuneOut = & python (Join-Path $root "benchmark.py") @tuneArgs 2>&1 | Out-String
        $tuneRc = $LASTEXITCODE
        $tuneOut | Out-File -Append $report -Encoding UTF8
        Log $tuneOut.TrimEnd()
        if ($tuneRc -eq 0) { Ok "arena tuning completed"; $tuningDone = $true }
        else { Ko "arena tuning did NOT complete (rc=$tuneRc)" }
    }
}

# ============================================================
# 5. RESUME TEST
# ============================================================
Hr "5. RESUME TEST (interruption + restart)"
$resumeDir = Join-Path $env:TEMP ("preflight_resume_" + $ts)
New-Item -ItemType Directory -Force -Path $resumeDir | Out-Null
# A tiny run: two cycles, then two more on top of the saved state.
$resumeEnv = [ordered]@{
    DAMA_WEIGHTS_DIR = $resumeDir; DAMA_CYCLES = "2"; DAMA_GAMES = "2"; DAMA_SIMS = "10"
    DAMA_ARENA_GAMES = "2"; DAMA_SPRT_MAX_GAMES = "2"; DAMA_SPRT_CHUNK = "2"
    DAMA_CHANNELS = "8"; DAMA_BLOCKS = "1"; DAMA_WORKERS = "1"; DAMA_STRENGTH_EVERY = "0"
}
# The script runs inside the caller's session, so these values are put back
# afterwards. Left in place, a .\run.ps1 launched right after from the same
# window would inherit an 8-channel network, four cycles and a weights folder
# that no longer exists.
$savedEnv = @{}
foreach ($k in $resumeEnv.Keys) {
    $savedEnv[$k] = [Environment]::GetEnvironmentVariable($k)
    [Environment]::SetEnvironmentVariable($k, $resumeEnv[$k])
}
try {
    & python (Join-Path $root "conductor.py") *> (Join-Path $resumeDir "run1.txt")
    $r1rc = $LASTEXITCODE
    $env:DAMA_CYCLES = "4"
    $out2 = & python (Join-Path $root "conductor.py") 2>&1 | Out-String
    $r2rc = $LASTEXITCODE
} finally {
    foreach ($k in $savedEnv.Keys) { [Environment]::SetEnvironmentVariable($k, $savedEnv[$k]) }
}
$out2 | Out-File (Join-Path $resumeDir "run2.txt")
Get-Content (Join-Path $resumeDir "run1.txt"), (Join-Path $resumeDir "run2.txt") | Out-File -Append $report -Encoding UTF8
if ($r1rc -eq 0 -and $r2rc -eq 0 -and $out2 -match "resumed the state" -and $out2 -match "\[cycle 003\]" -and $out2 -notmatch "\[cycle 001\]") {
    Ok "resume: the second run restarts from cycle 3 (not from 1)"
} else {
    Ko "resume: unexpected behavior (see report)"
}
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $resumeDir

# ============================================================
# 6. LibTorch backend of the C++ engine - a note, not a test
# ============================================================
Hr "6. LibTorch backend of the C++ engine"
# The ENGINE IS ASKED, nothing is declared. These lines used to be fixed text
# describing one particular machine: they said "there is no LibTorch, CMake or
# CUDA here" even where all three were present. Where they are present that text
# claims the production engine is fake when it is not, and whoever reads the
# report remotely has no way to notice.
# build.ps1 puts the engine under build\ with MSVC and CMake, next to the sources
# with g++: looking only in build\ declared a successful g++ build missing.
$engExe = @((Join-Path $root "engine_c\build\dama_engine.exe"),
            (Join-Path $root "engine_c\dama_engine.exe")) |
          Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $engExe) {
    Ko "the engine was not produced: without it, points 4 and 5 measured the Python path"
    Log "  Look at the outcome of point 1: the build did not get to the end."
} else {
    $probe = & $engExe --mode selfplay --n-games 1 --n-threads 1 --n-sims 2 --weights /nonexistent_.pt 2>&1 | Out-String
    if ("$probe" -match 'WITHOUT LibTorch') {
        Log "  The engine was built with the FAKE BACKEND only: it runs, but it does"
        Log "  not load the real network."
        Log ""
        Log "  Not to be confused: the C++ engine is complete and validated -- rules,"
        Log "  encoding, search, concurrency, self-play and arena are all verified at"
        Log "  point 1. What is missing is only the loading of the network."
        Log ""
        Log "  Consequence for the estimates: point 4 measured the Python path of this"
        Log "  machine, not the compiled engine. Between the two there are almost two"
        Log "  orders of magnitude, so do not use those numbers to size a production"
        Log "  run. Set LIBTORCH and launch again."
        Ok "engine present (fake backend: estimates not valid for production)"
    } else {
        Ok "engine built with the LibTorch backend: it is the one that will do the run"
        if ($env:DAMA_DEVICE -and $env:DAMA_DEVICE -ne "cpu") {
            Log "  With DAMA_DEVICE=$($env:DAMA_DEVICE) the estimates of point 4 are"
            Log "  those of the production configuration."
        } else {
            Log "  NOTE: DAMA_DEVICE is not set to a graphics card, so point 4"
            Log "  measured the processor. On the machine with the card launch"
            Log "  again with DAMA_DEVICE=cuda for the number that counts."
        }
    }
}

# ============================================================
# SUMMARY
# ============================================================
Hr "PREFLIGHT SUMMARY"
Log "  PASS = $($script:pass)    FAIL = $($script:fail)"
if ($summary) {
    Log ""
    Log "  Line to send back to whoever asked for the measurement:"
    Log "  $($summary.Trim())"
}
if (-not $tuningDone -and $env:DAMA_CPP_ENGINE) {
    Log ""
    Log "  The arena is two thirds of the cycle and its parameters are NOT tuned"
    Log "  on this machine. Once only, before the real run:"
    Log "    .\preflight.ps1 -TuneArena"
}
Log "  report: $report"
if ($script:fail -eq 0) {
    Log ""; Log ">>> PREFLIGHT PASSED."
    exit 0
} else {
    Log ""; Log ">>> PREFLIGHT FAILED ($($script:fail) checks) - do not start the run until they are all green."
    exit 1
}
