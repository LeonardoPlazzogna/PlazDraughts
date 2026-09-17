<#
run.ps1 - launch of the run on Windows, with checks BEFORE starting.
Mirror of run.sh, same philosophy: a wrong configuration must FAIL AT ONCE AND
LOUDLY, not degrade silently.

  .\run.ps1
  $env:DAMA_CYCLES = 100; .\run.ps1

It comes from the same mistake that motivated run.sh: a run launched with
`python conductor.py` without environment variables, hence with the defaults --
CPU device on a machine with a GPU, C++ engine disabled, workers fighting over
the cores. The conductor wrote it in the log and nobody noticed until the
process died.

DIFFERENCE FROM LINUX, and it must be said before someone loses a day over it:
on Windows build.ps1 WITHOUT LibTorch compiles the C++ engine with g++, so it
only has the fake backend -- useful to validate the pipeline, useless for
playing. For the neural network, set LIBTORCH before building: build.ps1 then
uses CMake, Visual Studio and LibTorch for Windows (see engine_c/README.md).
Without the real engine the run goes through the Python path: correct, much
slower, suited to development and analysis more than to a production run.
#>
[CmdletBinding()]
param(
    [switch]$Force,          # start even if the checks give warnings
    [int]$Retry = 0          # how many times to restart the run if the process dies
)

$ErrorActionPreference = 'Continue'
Set-Location -Path $PSScriptRoot

$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue) }
if (-not $py) { Write-Host "  X  no python interpreter found."; exit 1 }
$PY = $py.Source

# --- configuration (can be overridden from the environment) -------------------
# DAMA_DEVICE has no fixed value: it is decided after PROBING the card, a little
# further down. Before, the default was 'cpu', and whoever had a GPU had to
# remember to say so -- exactly the mistake this script exists to prevent.
if (-not $env:DAMA_WEIGHTS_DIR) { $env:DAMA_WEIGHTS_DIR = 'weights' }
if (-not $env:DAMA_CYCLES)      { $env:DAMA_CYCLES = '100' }

# The compiled engine, if present, must be used: it is two orders of magnitude
# faster than the interpreted path. The path is the one build.ps1 produces.
if (-not $env:DAMA_CPP_ENGINE) {
    $exeDef = Join-Path $PSScriptRoot 'engine_c\build\dama_engine.exe'
    if (Test-Path $exeDef) { $env:DAMA_CPP_ENGINE = $exeDef }
}

# ENGINE THREADS. The "usable" cores measured by saturation are not used: that
# measurement counts COMPUTE capacity, and this load is not limited by compute
# but by waiting for the answers of the card. The threads spend a lot of time
# idle, so having more of them than cores fills the gaps instead of contending.
#
# Going well above the number of usable cores can cut the cost per ply
# substantially in both self-play and the arena. It is the only lever that speeds
# things up WITHOUT touching how the engine plays -- it changes how many games
# run in parallel, not how each one is played; benchmark.py --sweep finds the
# value for a given machine.
if (-not $env:DAMA_CPP_THREADS) {
    $logical = [Environment]::ProcessorCount
    $env:DAMA_CPP_THREADS = [string][int][Math]::Max(4, [Math]::Floor($logical * 0.75))
}
# THE LOG LIVES INSIDE THE WEIGHTS FOLDER, not in the project folder.
#
# Two reasons, both found while handing a run over to someone else. The first:
# everything else a run produces lives there, and asking "send me the folder plus
# a file that lives somewhere else" is an invitation to forget it. The second,
# worse: 'run.log' was a FIXED path opened for appending, so two runs launched in
# different folders -- the right way to do two -- wrote their logs appended to
# the SAME file, mixed up and no longer attributable. Under the weights folder
# every run has its own, without having to think about it.
#
# The folder must be created here: Tee-Object opens the file when the pipeline
# starts, before the conductor has had a chance to create it.
if (-not (Test-Path $env:DAMA_WEIGHTS_DIR)) {
    New-Item -ItemType Directory -Force -Path $env:DAMA_WEIGHTS_DIR | Out-Null
}
$Log = if ($env:DAMA_LOG) { $env:DAMA_LOG }
       else { Join-Path $env:DAMA_WEIGHTS_DIR 'run.log' }

$warnings = 0
function Ok($m)    { Write-Host "  ok  $m" }
function Warn($m)  { Write-Host "  !   $m"; $script:warnings++ }
function Abort($m) { Write-Host ""; Write-Host "  X  $m"; Write-Host ""; Write-Host "Launch aborted."; exit 1 }

Write-Host "=== checks before the launch ==="

# 1. torch present and consistent with the requested device.
$t = & $PY -c "import torch;print(torch.__version__);print(torch.version.cuda)" 2>&1
if ($LASTEXITCODE -ne 0) { Abort "torch cannot be imported:`n      $t" }
$lines = @($t)
Ok "torch $($lines[0])"

# 2. The requested device is really usable.
#    is_available() is not enough: a real allocation is made, as on Linux. A
#    device declared and not working would make everything run on the CPU
#    silently.
$gpuProbe = @"
import torch
try:
    torch.zeros(8, device='cuda') * 2
    print('YES ' + torch.cuda.get_device_name(0))
except Exception as e:
    print('NO ' + type(e).__name__ + ': ' + str(e))
"@

# The right line is looked for AMONG THE LINES, instead of flattening everything
# into a single string. With "$g" -like 'YES *' a torch warning printed before
# the answer was enough -- and with a real GPU they do arrive -- for the check
# to fail and the script to reject a perfectly good card. A defect invisible
# without a usable card, where the NO branch is always taken.
function Get-GpuName($lines) {
    foreach ($l in @($lines)) { if ("$l".StartsWith('YES ')) { return "$l".Substring(4) } }
    return $null
}

# If nobody asked for a device, it is chosen BY LOOKING: is there a usable card?
# then it is used. The default value 'cpu' forced one to remember to say 'cuda',
# and forgetting costs orders of magnitude without anything stopping.
if (-not $env:DAMA_DEVICE) {
    $name = Get-GpuName (& $PY -c $gpuProbe 2>&1)
    if ($name) {
        $env:DAMA_DEVICE = 'cuda'
        Ok "device chosen automatically: cuda ($name)"
    } else {
        $env:DAMA_DEVICE = 'cpu'
        Ok "device chosen automatically: cpu (no usable GPU)"
    }
}

if ($env:DAMA_DEVICE -ne 'cpu') {
    $g = & $PY -c $gpuProbe 2>&1
    $name = Get-GpuName $g
    if ($name) {
        Ok "GPU $name"
    } else {
        Abort "requested device '$($env:DAMA_DEVICE)' but the GPU is not usable.
      $g
      On this machine torch might be the CPU-only build: check with
        python -c ""import torch;print(torch.version.cuda)""
      If it prints None, install a build with CUDA or launch on the CPU:
        `$env:DAMA_DEVICE='cpu'; .\run.ps1"
    }
} else {
    # The opposite, and it is THE MISTAKE THAT GAVE BIRTH TO THIS SCRIPT: running
    # on the CPU with a usable card sitting idle next to it. The conductor writes
    # it in the log and nobody notices. Here it is checked and blocked.
    $g = & $PY -c $gpuProbe 2>&1
    $name = Get-GpuName $g
    if ($name) {
        Warn "device set to CPU, but this machine has a usable GPU: $name
      On the CPU the same run is orders of magnitude slower. To use the GPU:
        `$env:DAMA_DEVICE='cuda'; .\run.ps1"
    } else {
        Ok "device: cpu (no usable GPU on this machine)"
    }
}

# 3. C++ engine: if present, it must be able to load a network AND know the
#    options the conductor passes to it. A binary older than the Python code
#    would make every cycle fall back to the slow path, printing a line that
#    gets lost.
if ($env:DAMA_CPP_ENGINE) {
    if (-not (Test-Path $env:DAMA_CPP_ENGINE)) {
        # The path without .exe is the most likely mistake: it is what one
        # writes out of habit, copying it from the Linux instructions.
        $withExe = "$($env:DAMA_CPP_ENGINE).exe"
        if (Test-Path $withExe) {
            Abort "C++ engine not found: $($env:DAMA_CPP_ENGINE)
      But $withExe exists -- on Windows the extension is needed:
        `$env:DAMA_CPP_ENGINE='$withExe'"
        }
        Abort "C++ engine not found: $($env:DAMA_CPP_ENGINE)
      Build it with:  cd engine_c; .\build.ps1
      Or launch without it:  `$env:DAMA_CPP_ENGINE=''; .\run.ps1"
    }
    $probe = & $env:DAMA_CPP_ENGINE --mode selfplay --n-games 1 --n-threads 1 --n-sims 2 --weights /nonexistent_.pt 2>&1
    if ("$probe" -match 'WITHOUT LibTorch') {
        Warn "the C++ engine is built WITHOUT LibTorch: it only has the fake backend.
      The samples produced would have NO playing value. build.ps1 produces it
      when LIBTORCH is not set: set it and rebuild (cd engine_c; .\build.ps1)."
    } else {
        Ok "C++ engine: can load a network"
    }
    $help = & $env:DAMA_CPP_ENGINE --help 2>&1
    foreach ($opt in @('--temp-plies', '--c-puct-b', '--n-sims-b')) {
        if ("$help" -notmatch [regex]::Escape($opt)) {
            Abort "the C++ engine is OLDER than the Python code: it does not know
      $opt, which the conductor passes to it. It would fall back to the Python
      path at every cycle, without anything shouting it. Rebuild it: cd engine_c; .\build.ps1"
        }
    }
    Ok "engine up to date (knows the options the conductor uses)"

    # THE FALLBACK. With the compiled engine the process pool is not created, and
    # the native Windows crash cannot happen. But if the engine breaks even for a
    # single cycle the conductor falls back to the Python path, and creates that
    # pool with DAMA_WORKERS -- that is, fifteen processes with the default
    # values, exactly the configuration that dies on Windows. Defending against
    # one crash by switching to a worse one, at night to boot, with nobody
    # watching.
    $nwFallback = if ($env:DAMA_WORKERS) {
        [int]$env:DAMA_WORKERS
    } else {
        [Math]::Min([Math]::Max(1, [Environment]::ProcessorCount - 1), 64)
    }
    if ($nwFallback -gt 4) {
        Write-Host "  i   if the engine broke, the fallback would use $nwFallback Python"
        Write-Host "      processes: on Windows that is the configuration at risk of a native"
        Write-Host "      crash. It costs nothing to be safe:  `$env:DAMA_WORKERS='4'"
        Write-Host "      (with the engine active the pool is not even created)"
    }
} else {
    # "i" and not "!": this line does NOT increase the warning counter, and with
    # the same marker as the lines that do increase it the total at the bottom
    # ("N warning(s)") looked off by one.
    Write-Host "  i   C++ engine disabled: self-play through the Python path (much slower)"
}

# 4. Dry run left on: it produces samples with no playing value, and the defect
#    does not show in the metrics because the loss goes down anyway.
if ($env:DAMA_CPP_FAKE -and $env:DAMA_CPP_FAKE -ne '0') {
    Abort "DAMA_CPP_FAKE is on: the engine would run WITHOUT a network and the samples
      would have no playing value. Turn it off:  `$env:DAMA_CPP_FAKE=''"
}

# 5. The crash of parallel self-play on the CPU, which is a WINDOWS defect and
#    so is checked here and not in run.sh.
#
#    With the CPU device, no C++ engine and the default number of workers
#    (cores-1, fifteen on a 16-core machine) the run can die of an "access
#    violation" inside the forward pass of the network -- a NATIVE crash, which
#    Python cannot catch: the process simply vanishes. Isolated with
#    PYTHONFAULTHANDLER=1; the exact point changes every time, a sign of memory
#    corruption and not of a logic error. It is not deterministic: with few
#    cycles it sometimes passes, which is why it survived for so long. Tried
#    without success: limiting the PyTorch threads, KMP_DUPLICATE_LIB_OK. Tried
#    WITH success: a smaller pool.
#
#    It is a warning and not a stop because it is probabilistic: whoever wants
#    to take the risk does so with -Force. Better to know it now, though, than
#    after six hours of computation thrown away.
#
#    IT APPLIES ON THE GPU TOO, and before, this check did not notice. Without
#    the C++ engine the conductor creates the process pool in both cases: on the
#    CPU because it is the only way to parallelize, on the GPU because the
#    workers stay CPU processes doing search and encoding while a server holds
#    the card. The pool -- that is, the ingredient of the crash -- is there
#    anyway.
$pythonPool = (-not $env:DAMA_CPP_ENGINE)
if ($pythonPool) {
    $nw = if ($env:DAMA_WORKERS) {
        [int]$env:DAMA_WORKERS
    } else {
        [Math]::Min([Math]::Max(1, [Environment]::ProcessorCount - 1), 64)
    }
    $where = if ($env:DAMA_DEVICE -eq 'cpu') { "on the CPU" } else { "with the network on $($env:DAMA_DEVICE)" }
    if ($nw -gt 4) {
        Warn "parallel self-play $where with $nw worker processes: on Windows
      it is the configuration that crashes natively during training (see
      docs/running.md). The pool exists with the graphics card too, so the risk
      does not go away. The only remedy found is a smaller pool:
        `$env:DAMA_WORKERS='4'; .\run.ps1"
    } else {
        Ok "$nw worker processes (within the safe limit on Windows)"
    }
}

# 6. Previous state: if present, the run RESUMES. It must be said before, not after.
$state = Join-Path $env:DAMA_WEIGHTS_DIR 'train_state.pkl'
if (Test-Path $state) {
    $c = & $PY -c "import pickle,sys;print(pickle.load(open(sys.argv[1],'rb'))['cycle'])" $state 2>&1
    if ($LASTEXITCODE -eq 0) {
        Ok "state found: resumes from cycle $([int]$c + 1) (folder $($env:DAMA_WEIGHTS_DIR))"
    } else {
        Warn "state present but unreadable: the run would restart from cycle 1"
    }
} else {
    Ok "no previous state: starting from cycle 1"
}

if ($warnings -gt 0 -and -not $Force) {
    Write-Host ""
    Write-Host "  $warnings warning(s). To start anyway:  .\run.ps1 -Force"
    exit 1
}

$games = if ($env:DAMA_GAMES) { $env:DAMA_GAMES } else { '400' }
Write-Host ""
Write-Host "=== start: $($env:DAMA_CYCLES) cycles, device=$($env:DAMA_DEVICE), log in $Log ==="
Write-Host "    games/cycle=$games"
Write-Host "    follow with:  .\watch.ps1 -Follow"
Write-Host ""

# -u turns off buffering: without it, the log arrives in chunks and looks stuck.
# Python's exit code survives the pipeline because Tee-Object is a cmdlet and not
# a native command: $LASTEXITCODE stays the one of python.
#
# AUTOMATIC RESTART (-Retry N). On Windows the process pool can die of a NATIVE
# crash: the process vanishes without an exception and without a Python trace.
# On a long unattended run this costs all the time between the crash and the
# moment someone notices -- days, if it happens at night.
#
# Restarting is safe because the state is written AT EVERY CYCLE, atomically: at
# most the cycle in progress is lost, and the conductor resumes from the next
# one. It is not a way of HIDING the crash -- every restart is printed and ends
# up in the log, where the dashboard counts it among the anomalies.
#
# The protection against an endless loop is not the number of attempts but the
# DURATION: if the conductor dies in less than a minute it is not meeting the
# rare crash, it is rejecting the configuration, and retrying a thousand times
# would not fix it.
$attempts = 0
while ($true) {
    $start = Get-Date
    & $PY -u conductor.py 2>&1 | Tee-Object -FilePath $Log -Append
    $rc = $LASTEXITCODE
    if ($rc -eq 0) { break }

    $duration = ((Get-Date) - $start).TotalSeconds
    Write-Host ""
    Write-Host "[run.ps1] the conductor exited with status $rc after $([int]$duration)s (see $Log)"
    if ($Retry -le 0) {
        Write-Host "[run.ps1] automatic restart disabled. To enable it:  .\run.ps1 -Retry 20"
        break
    }

    # INTENTIONAL STOP. Without this, whoever presses Ctrl-C on the seventh day to
    # close the run would see it restart: from the outside an intentional stop and
    # a crash look too much alike. Two ways of saying "enough":
    #
    #   1. the file STOP in the weights folder -- it also works remotely, and it
    #      can be created while the run is going, without hurry;
    #   2. the code with which Windows marks a process interrupted by Ctrl-C. It
    #      is not reliable on its own (interrupted at an arbitrary point, Python
    #      can exit with 1 like any error), which is why point 1 exists.
    $stopFile = Join-Path $env:DAMA_WEIGHTS_DIR 'STOP'
    if (Test-Path $stopFile) {
        Write-Host "[run.ps1] found $($stopFile): intentional stop, not restarting."   # $($x) and not $x: after a variable the colon opens a
                                                                          # scope qualifier (as in $env:NAME) and the line blows up
                                                                          # at run time -- the syntax check does not see it.
        Remove-Item $stopFile -Force -ErrorAction SilentlyContinue
        break
    }
    if ($rc -eq -1073741510 -or $rc -eq 3221225786) {
        Write-Host "[run.ps1] interrupted from the keyboard: not restarting."
        break
    }
    if ($duration -lt 60) {
        Write-Host "[run.ps1] died in less than a minute: it is not an occasional crash,"
        Write-Host "          it is the configuration. Not retrying."
        break
    }
    if ($attempts -ge $Retry) {
        Write-Host "[run.ps1] the $Retry restart attempts are used up."
        break
    }
    $attempts++
    $msg = "[run.ps1] RESTART $attempts/$Retry in 15s (resumes from the cycle after the last one saved)"
    Write-Host $msg
    # Through Tee-Object, like the conductor's output, and not Out-File. In
    # Windows PowerShell 5.1 Tee-Object writes the log in UTF-16, and a line
    # appended as UTF-8 came out unreadable in the middle of it: the dashboard,
    # which counts restarts, never saw one, and a line of odd length garbled
    # everything written after it.
    $msg | Tee-Object -FilePath $Log -Append | Out-Null
    Start-Sleep -Seconds 15
}
if ($attempts -gt 0) {
    Write-Host "[run.ps1] the run needed $attempts restart(s): look for them in the log."
}
exit $rc
