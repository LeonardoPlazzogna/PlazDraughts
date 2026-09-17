@echo off
rem launch.cmd - entry point on Windows when the project comes from outside.
rem
rem   launch.cmd                 checks and start of the run
rem   launch.cmd -Force          starts even with warnings
rem   launch.cmd -Retry 20       restarts by itself if the process dies
rem   launch.cmd watch           dashboard
rem   launch.cmd watch -Follow   dashboard that refreshes by itself
rem   launch.cmd preflight       complete validation before starting
rem   launch.cmd preflight -TuneArena
rem                             looks for the fastest parameters for the arena,
rem                             which is two thirds of the cycle (once only)
rem   launch.cmd build           builds the C++ engine
rem   launch.cmd benchmark --device cuda
rem                             how long the complete run would take here
rem
rem WHY IT EXISTS. Windows refuses to run PowerShell scripts that came from the
rem internet: the default policy is RemoteSigned, and a file extracted from a
rem downloaded archive carries the "comes from the network" mark. The message
rem that comes out talks about disabled scripts or a missing digital signature,
rem and sends one looking for the wrong solution -- usually changing the policy
rem of the whole machine, which is a permanent security change and is not needed.
rem
rem Here it is worked around in the narrow way: -ExecutionPolicy Bypass applies
rem ONLY to this process, does not touch the system settings and leaves no trace
rem after closing. -NoProfile prevents the user's personal profile from changing
rem environment variables under the run's feet.
rem
rem TWO BATCH TRAPS, met while writing this very file:
rem
rem   1. inside a parenthesized block the arguments are expanded BEFORE shift is
rem      executed, so "if ... ( shift & prog %1 )" still passes the old
rem      argument. That is why labels are used here.
rem   2. %* is NOT affected by shift: after removing the subcommand it would
rem      still return the full list. And writing %1 %2 %3 silently truncates
rem      beyond that number -- with "benchmark --device cuda --weights ... --sims 400"
rem      the options at the end were lost, and the measurement ran with values
rem      different from the requested ones without saying so. The arguments are
rem      therefore collected in a loop, one by one, until they run out.
setlocal
set "HERE=%~dp0"
set "PS=powershell -NoProfile -ExecutionPolicy Bypass -File"

if /i "%~1"=="watch"     goto :watch
if /i "%~1"=="preflight" goto :preflight
if /i "%~1"=="build"     goto :build
if /i "%~1"=="benchmark" goto :benchmark

%PS% "%HERE%run.ps1" %*
goto :eof

:watch
call :rest %*
%PS% "%HERE%watch.ps1" %REST%
goto :eof

:preflight
call :rest %*
%PS% "%HERE%preflight.ps1" %REST%
goto :eof

:build
call :rest %*
%PS% "%HERE%engine_c\build.ps1" %REST%
goto :eof

rem The benchmark is Python, not PowerShell: it is called directly, without
rem going through the script execution policy.
:benchmark
call :rest %*
python "%HERE%benchmark.py" %REST%
goto :eof

rem Collects everything except the first argument (the subcommand). %1 is used
rem and not %~1 so as not to lose the quotes of a path with spaces.
:rest
set "REST="
shift
:rest_loop
if "%~1"=="" goto :eof
set "REST=%REST% %1"
shift
goto :rest_loop
