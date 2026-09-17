# Builds and validates the whole C++ engine, on Windows.
#
#   .\build.ps1                          g++, all the tests, fake backend
#   $env:LIBTORCH = 'C:\libtorch'
#   .\build.ps1                          the real engine (CMake + MSVC + LibTorch)
#   .\build.ps1 -Libtorch C:\libtorch    the same, without the variable
#
# TWO PATHS, and the difference decides what the resulting binary is for:
#
#   g++      : builds everything and runs the validation tests, but the engine
#              has ONLY THE FAKE BACKEND -- it evaluates positions with made-up
#              numbers. It serves to check rules, search and concurrency; NOT
#              to play or to produce self-play samples of any value.
#   LibTorch : the real engine, the one that loads the network. It needs CMake,
#              Visual Studio with the C++ tools, and LibTorch for Windows.
#
# The LibTorch path follows step by step the working Windows build of the chess
# project, including the three things that cost time there: where the
# executable ends up, the DLLs next to it, and the handling of standard error
# (see below).
param(
    [string]$Libtorch = $(if ($env:LIBTORCH) { $env:LIBTORCH } else { $env:DAMA_LIBTORCH })
)

# "Continue" and NOT "Stop", and it is not an oversight.
#
# Native commands write warnings and diagnostics to standard error. When this
# script's output is REDIRECTED -- which is what preflight.ps1 does --
# PowerShell 5.1 wraps every standard-error line in an ErrorRecord; with
# ErrorActionPreference='Stop' that ErrorRecord becomes a FATAL error even if the
# command exited with 0. The symptom is baffling: the build run by hand works,
# the same build inside the preflight "fails" with nothing broken. So the EXIT
# CODES are relied upon, and they tell the truth.
#
# For the same reason $? is never used here after a native command: under
# redirection it becomes false even when the compilation went perfectly well.
$ErrorActionPreference = "Continue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# A missing tool is not an exception: it is something to read and fix.
function Fail($m) { Write-Host ""; Write-Host "  X  $m"; Write-Host ""; exit 1 }

# --- LibTorch path: CMake + MSVC ----------------------------------------------
if ($Libtorch) {
    # Not "the folder exists" but "the folder is a LibTorch usable by CMake".
    # It tells apart the three real mistakes -- wrong path, archive not
    # extracted, a variant without the CMake files -- which otherwise all show
    # up as the same wall of CMake errors one screen later.
    if (-not (Test-Path (Join-Path $Libtorch 'share\cmake\Torch'))) {
        Fail "'$Libtorch' does not contain share\cmake\Torch: wrong path,
      archive not extracted, or a LibTorch variant without the CMake
      files."
    }
    # Wrong variant: the easiest mistake to make, because the download page
    # offers Linux and Windows side by side and the folders look identical.
    # Without this check CMake configures, compiles, and fails much later with
    # incomprehensible link errors.
    if (@(Get-ChildItem (Join-Path $Libtorch 'lib') -Filter *.dll -ErrorAction SilentlyContinue).Count -eq 0) {
        Fail "'$Libtorch\lib' contains no DLLs: this is the LibTorch for
      LINUX. The Windows one is needed, Release variant, built with MSVC
      (MinGW does not work: the libraries are built with MSVC)."
    }
    # Visual Studio must be LOOKED FOR, not expected on the PATH.
    #
    # Whoever installs the build tools gets cmake, ninja and the compiler inside
    # the Visual Studio folder, not on the PATH of a normal PowerShell: they
    # only work from the "Developer PowerShell". Stopping with "install CMake"
    # for someone who already has CMake is a perfect way to make them give up.
    # Verified in an installation where all three are present and none of them
    # is on the PATH.
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $vsroot = $null
    if (Test-Path $vswhere) {
        $vsroot = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null | Select-Object -First 1)
    }

    $cmake = (Get-Command cmake -ErrorAction SilentlyContinue)
    $cmakeExe = if ($cmake) { $cmake.Source } else { $null }
    if (-not $cmakeExe -and $vsroot) {
        $c = Join-Path $vsroot 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
        if (Test-Path $c) { $cmakeExe = $c }
    }
    if (-not $cmakeExe) {
        Fail "cmake not found, neither on the PATH nor inside Visual Studio.
      Install it (winget install Kitware.CMake) or run without
      -Libtorch for the engine with the fake backend only."
    }

    # The compiler environment. Without it cmake configures and then cannot find cl.exe.
    $vcvars = $null
    if ($vsroot) {
        $v = Join-Path $vsroot 'VC\Auxiliary\Build\vcvars64.bat'
        if (Test-Path $v) { $vcvars = $v }
    }
    if (-not $vcvars -and -not (Get-Command cl -ErrorAction SilentlyContinue)) {
        Fail "MSVC compiler not found. The Visual Studio 'Build Tools for C++'
      are needed. LibTorch for Windows is built with MSVC:
      MinGW is not compatible."
    }

    Write-Host "visual studio: $(if ($vsroot) { $vsroot } else { 'already in the environment' })"
    Write-Host "cmake:         $cmakeExe"
    Write-Host "libtorch:      $Libtorch"

    # LibTorch and Python's torch must MATCH (version and CUDA tag): the engine
    # loads a TorchScript exported by that torch. A warning and not a stop,
    # because small gaps often work and it is not worth blocking someone who
    # knows what they are doing.
    $bv = Join-Path $Libtorch 'build-version'
    if (Test-Path $bv) {
        $vLib = (Get-Content $bv -TotalCount 1).Trim()
        $vPy = (& python -c "import torch;print(torch.__version__)" 2>$null)
        if ($LASTEXITCODE -eq 0 -and $vPy -and "$vPy".Trim() -ne $vLib) {
            Write-Host "  !   LibTorch $vLib against torch $($vPy.Trim()): different versions."
            Write-Host "      The engine loads a TorchScript exported by Python's torch;"
            Write-Host "      if loading fails, this is the most likely cause."
        } else {
            Write-Host "version: $vLib (consistent with Python's torch)"
        }
    }

    $build = Join-Path $here 'build'

    # The two commands run INSIDE the compiler environment, opened by
    # vcvars64.bat. Not a detail: without it cmake configures, then cannot find
    # cl.exe and fails halfway. They go through a temporary batch file because
    # vcvars sets dozens of variables that must survive until the compilation,
    # and PowerShell does not inherit the environment of a child process.
    $script = Join-Path ([IO.Path]::GetTempPath()) ("dama_build_" + [guid]::NewGuid().ToString('N') + ".bat")

    # THE PATHS DO NOT GO INTO THE BATCH FILE; the names of the variables that
    # hold them do. Written into the file, they would have to survive two
    # encodings that do not talk to each other -- the one PowerShell writes with
    # and the code page cmd.exe reads with -- and the second changes with the
    # context the script is launched from. It really happened: the same build
    # succeeded from one terminal and failed when the same script was called
    # from another, with 'cannot find the path' on a folder that existed,
    # because an accented letter in the path had become a different byte.
    # Environment variables reach the child process in Unicode and do not go
    # through the file: so the file stays pure ASCII and the encoding stops
    # mattering.
    $env:DAMA_B_HERE  = $here
    $env:DAMA_B_BUILD = $build
    $env:DAMA_B_LT    = $Libtorch
    $env:DAMA_B_CMAKE = $cmakeExe
    $env:DAMA_B_VCV   = $vcvars
    # vcvars64.bat looks for vswhere.exe, and with the Build Tools it is not on
    # the PATH: seen failing exactly like that. We already know where it is.
    $inst = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer'

    $lines = @('@echo off')
    if (Test-Path $inst) { $env:DAMA_B_INST = $inst; $lines += 'set "PATH=%DAMA_B_INST%;%PATH%"' }
    if ($vcvars) { $lines += 'call "%DAMA_B_VCV%" >nul || exit /b 1' }
    $lines += 'cd /d "%DAMA_B_HERE%" || exit /b 1'
    # --config Release is needed with the Visual Studio generator, which is
    # multi-configuration and ignores CMAKE_BUILD_TYPE: without it, it builds in
    # Debug, i.e. an engine many times slower and nothing to flag it.
    $lines += '"%DAMA_B_CMAKE%" -B "%DAMA_B_BUILD%" -S "%DAMA_B_HERE%" -DWITH_LIBTORCH=ON -DCMAKE_PREFIX_PATH="%DAMA_B_LT%" -DCMAKE_BUILD_TYPE=Release || exit /b 1'
    $lines += '"%DAMA_B_CMAKE%" --build "%DAMA_B_BUILD%" --config Release || exit /b 1'
    # The file is now pure ASCII, so any encoding works; Oem stays because it is
    # what cmd.exe expects.
    Set-Content -Path $script -Value $lines -Encoding Oem

    Write-Host ""
    Write-Host "== configure and build (MSVC, Release) =="
    & cmd /c "`"$script`""
    $rc = $LASTEXITCODE
    Remove-Item $script -Force -ErrorAction SilentlyContinue
    if ($rc -ne 0) {
        Fail "build failed (exit $rc). See the output above.
      Check that LibTorch is the Windows variant built with MSVC and
      that it matches the installed CUDA version."
    }

    # Location of the executable, NORMALIZED. The Visual Studio generator puts
    # it in build\Release\, the Ninja one in build\: without this copy the path
    # to put in DAMA_CPP_ENGINE would depend on the generator, i.e. it would
    # change from machine to machine.
    $found = Get-ChildItem -Path $build -Recurse -Filter 'dama_engine.exe' -ErrorAction SilentlyContinue |
             Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $found) { Fail "built, but dama_engine.exe cannot be found under $build" }
    $exe = Join-Path $build 'dama_engine.exe'
    if ($found.FullName -ne $exe) { Copy-Item $found.FullName $exe -Force }

    # DLLs next to the executable. CMakeLists already copies them where the
    # compiler writes, but if the executable was moved above they are ALSO
    # needed here: Windows has no rpath, the loader only looks in the
    # executable's folder and on the PATH. Without them the exe dies before
    # main with 0xc0000135 and no message.
    $libdir = Join-Path $Libtorch 'lib'
    Copy-Item (Join-Path $libdir '*.dll') $build -Force -ErrorAction SilentlyContinue
    $rel = Join-Path $build 'Release'
    if (Test-Path $rel) { Copy-Item (Join-Path $libdir '*.dll') $rel -Force -ErrorAction SilentlyContinue }

    # The binary MUST start. This check turns the worst failure -- a process
    # that vanishes without a word in the middle of a run -- into an error line
    # now.
    $helpOut = & $exe --help 2>&1
    if ($LASTEXITCODE -ne 0) {
        Fail "dama_engine.exe was built but does NOT START (exit $LASTEXITCODE).
      If it printed nothing, DLLs are missing: check that
      $libdir contains the .dll files and that they are the Windows ones.
      $helpOut"
    }
    # And it must be able to load a network. The binary is asked by trying to
    # load a file that does not exist: the error that comes back says whether
    # the LibTorch backend is compiled in (--help does not say it).
    $probe = & $exe --mode selfplay --n-games 1 --n-threads 1 --n-sims 2 --weights /nonexistent_.pt 2>&1
    if ("$probe" -match 'WITHOUT LibTorch') {
        Fail "built, but the binary declares it does NOT have LibTorch:
      -DWITH_LIBTORCH=ON had no effect. Delete $build and try again."
    }

    Write-Host ""
    Write-Host "BUILD OK with the LibTorch backend: $exe"
    Write-Host "use it with:  `$env:DAMA_CPP_ENGINE='$exe'"
    Write-Host "before producing results, run parity_check (see README.md)."
    exit 0
}

# --- g++ path: everything except the LibTorch backend -------------------------
$gpp = (Get-Command g++ -ErrorAction SilentlyContinue)
if (-not $gpp) { Fail "g++ not found on the PATH (a C++20 compiler is needed)." }
Write-Host "g++: $($gpp.Source)"

function Build-And-Run($name, $runArgs) {
    Write-Host ""
    Write-Host "--- $name ---"
    & g++ -std=c++20 -O2 -march=native -pthread "$here\$name.cpp" -o "$here\$name.exe"
    if ($LASTEXITCODE -ne 0) { Fail "build of $name failed" }
    if ($runArgs) { & "$here\$name.exe" @runArgs } else { & "$here\$name.exe" }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Build-And-Run "perft"
Build-And-Run "encoder_parity"
Build-And-Run "mcts_selfplay_test"
Build-And-Run "threading_test" @("8", "150", "256")
Build-And-Run "selfplay_mt_test" @("6", "4", "50", "$here\selfplay_test.bin")
Build-And-Run "arena_test" @("12", "4", "30")

# main executable (not a test: it is only compiled)
Write-Host ""
Write-Host "--- dama_engine ---"
& g++ -std=c++20 -O2 -march=native -pthread "$here\main.cpp" -o "$here\dama_engine.exe"
if ($LASTEXITCODE -ne 0) { Fail "build of dama_engine failed" }
Write-Host "built: $here\dama_engine.exe (use --help for the options)"
Write-Host ""
Write-Host "WARNING: this binary has only the FAKE BACKEND -- it evaluates"
Write-Host "positions with made-up numbers. Fine for the tests, NOT for playing"
Write-Host "or for generating samples. For the real engine:"
Write-Host "    `$env:LIBTORCH='C:\path\to\libtorch'; .\build.ps1"
exit 0
