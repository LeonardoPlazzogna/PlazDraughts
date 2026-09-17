<#
watch.ps1 - dashboard of the run on Windows. Mirror of watch.sh: it reads
metrics.csv, ladder.json and run.log and shows progress, trend and estimated
end.

  .\watch.ps1              a snapshot
  .\watch.ps1 -Follow      refreshes every 60 s
  $env:DAMA_WEIGHTS_DIR='w_A'; $env:DAMA_CYCLES=100; .\watch.ps1

Dependencies: none beyond PowerShell. The bash version relies on awk and on
python to read the dates; here that is not needed, and it is better this way:
the columns are read by NAME instead of by position, so adding a column in the
middle of the CSV does not silently shift all the others -- which is how a
dashboard starts lying without anyone noticing.
#>
[CmdletBinding()]
param(
    [switch]$Follow
)

Set-Location -Path $PSScriptRoot

$Dir = if ($env:DAMA_WEIGHTS_DIR) { $env:DAMA_WEIGHTS_DIR } else { 'weights' }
$Tot = if ($env:DAMA_CYCLES)      { [int]$env:DAMA_CYCLES }  else { 100 }
# The same default as run.ps1, which writes the log INSIDE the weights folder.
# With 'run.log' in the project folder the dashboard looked for a file nobody
# writes, and always reported "no anomalies".
$Log = if ($env:DAMA_LOG)         { $env:DAMA_LOG }          else { Join-Path $Dir 'run.log' }
$Window = if ($env:DAMA_PACE_WINDOW) { [int]$env:DAMA_PACE_WINDOW } else { 20 }
$Every = if ($env:DAMA_WATCH_EVERY) { [int]$env:DAMA_WATCH_EVERY } else { 60 }
$Csv = Join-Path $Dir 'metrics.csv'

# The conductor writes the dates in UTC, but the oldest rows might not carry the
# time zone. AssumeUniversal+AdjustToUniversal treats them all the same way:
# without it, a row with a time zone and one without would be read against
# different references and the difference between the two would be off by hours.
function Read-Date($s) {
    $style = [System.Globalization.DateTimeStyles]::AssumeUniversal -bor `
             [System.Globalization.DateTimeStyles]::AdjustToUniversal
    $out = [datetime]::MinValue
    if ([datetime]::TryParse($s, [System.Globalization.CultureInfo]::InvariantCulture,
                             $style, [ref]$out)) { return $out }
    return $null
}

function Num($v, $d) {
    if ($null -eq $v -or "$v" -eq '') { return $d }
    $out = 0.0
    if ([double]::TryParse("$v", [System.Globalization.NumberStyles]::Float,
                           [System.Globalization.CultureInfo]::InvariantCulture, [ref]$out)) {
        return $out
    }
    return $d
}

function Show {
    Write-Host ""
    Write-Host ("===== {0} =====" -f (Get-Date -Format 'dd/MM HH:mm:ss'))

    if (-not (Test-Path $Csv)) {
        Write-Host "$Csv does not exist yet: the run has not completed its first cycle."
        if (Test-Path $Log) {
            Write-Host ""
            Write-Host "--- tail of $Log ---"
            Get-Content $Log -Tail 15
        }
        return
    }

    $rows = @(Import-Csv $Csv | Where-Object { $_.cycle -match '^\d+$' })
    if ($rows.Count -eq 0) {
        Write-Host "$Csv has no cycle rows yet."
        return
    }

    Write-Host "--- latest cycles ---"
    Write-Host " cycle  loss     p     v  arena pr  entr  qspr  vmae draw%   min"
    foreach ($r in ($rows | Select-Object -Last 15)) {
        Write-Host ("{0,6} {1,5:F3} {2,5:F3} {3,5:F3}  {4,5:F2} {5,2} {6,5:F3} {7,5:F3} {8,5:F3} {9,5:F2} {10,5:F1}" -f `
            [int]$r.cycle, (Num $r.loss 0), (Num $r.loss_policy 0), (Num $r.loss_value 0),
            (Num $r.arena_score 0), [int](Num $r.promoted 0), (Num $r.entropy_mean 0),
            (Num $r.q_spread_mean 0), (Num $r.value_mae 0), (Num $r.draw_rate 0),
            ((Num $r.duration_s 0) / 60))
    }

    # strength against fixed opponents: sparse rows (every DAMA_STRENGTH_EVERY cycles)
    $strength = @($rows | Where-Object { "$($_.strength)" -ne '' })
    if ($strength.Count -gt 0) {
        Write-Host ""
        Write-Host "--- strength against fixed opponents ---"
        foreach ($r in ($strength | Select-Object -Last 6)) {
            $e = if ("$($r.elo)" -eq '') { '-' } else { $r.elo }
            Write-Host ("cycle {0,4}  elo_gen={1,-8} {2}" -f [int]$r.cycle, $e, $r.strength)
        }
    }

    $ladder = Join-Path $Dir 'ladder.json'
    if (Test-Path $ladder) {
        Write-Host ""
        Write-Host "--- Elo across generations ---"
        try {
            $entries = @(Get-Content $ladder -Raw | ConvertFrom-Json)
            foreach ($v in ($entries | Select-Object -Last 6)) {
                Write-Host ("  gen {0,3}  (cycle {1,4})  Elo {2,8:+0.0;-0.0; 0.0}" -f `
                    [int]$v.gen, [int]$v.cycle, [double]$v.elo)
            }
        } catch {
            Write-Host "  ladder.json unreadable"
        }
    }

    Write-Host ""
    Write-Host "--- progress ---"
    # The pace is measured from the TIMES of the rows, not from the duration
    # column. Two reasons, and the second is the one that matters:
    #
    #   1. the duration_s column long excluded the evaluation phase, so the
    #      estimates came out short. It is correct now, but the CSVs already
    #      written stay wrong;
    #   2. the times ALSO include the time that ends up in no measured phase --
    #      saving the resume buffer, I/O waits, the GPU shared with something
    #      else. It is the time that really passes, which is exactly what is
    #      needed to say when it ends.
    #
    # Mean over the last 20 intervals, and the window is not arbitrary: the
    # strength measurements fire every DAMA_STRENGTH_EVERY cycles (20 by default)
    # and cost several times a normal cycle. A window of 10 contains that cycle
    # only half of the time, so the estimate would swing between one reading and
    # the next with nothing changed. A window equal to the period always contains
    # exactly one.
    #
    # Nor is the mean from the start of the run good, which has the opposite
    # defect: the cycle gets longer as the network improves, because the games
    # get longer, and a global mean systematically underestimates the future.
    # Both are printed: if they diverge a lot, the run is slowing down.
    $last = [int]$rows[-1].cycle
    $t = @()
    foreach ($r in $rows) {
        $d = Read-Date $r.timestamp
        if ($null -ne $d) { $t += $d }
    }
    $gaps = @()
    for ($i = 1; $i -lt $t.Count; $i++) {
        $g = ($t[$i] - $t[$i - 1]).TotalSeconds
        if ($g -gt 0) { $gaps += $g }
    }
    $global = 0.0
    if ($gaps.Count -gt 0) {
        $global = ($gaps | Measure-Object -Average).Average
    }
    $recent = @($gaps | Select-Object -Last $Window)
    if ($recent.Count -gt 0) {
        $m = ($recent | Measure-Object -Average).Average
    } else {
        # a single row: fall back on the measured phases
        $r = $rows[-1]
        $m = (Num $r.t_selfplay 0) + (Num $r.t_train 0) + (Num $r.t_arena 0) + (Num $r.t_eval 0)
        if ($m -le 0) { $m = Num $r.duration_s 0 }
    }
    $left = [Math]::Max(0, $Tot - $last)
    $pct = [int](100 * $last / [Math]::Max(1, $Tot))
    Write-Host ("cycle {0}/{1} ({2}%)   {3:F1} min/cycle (last {4};  since the start {5:F1})" -f `
        $last, $Tot, $pct, ($m / 60), $recent.Count, ($global / 60))
    if ($left -gt 0) {
        $end = (Get-Date).AddSeconds($left * $m)
        Write-Host ("{0} cycles left = {1:F1} h   ->   estimated end ~ {2}" -f `
            $left, ($left * $m / 3600), $end.ToString('dd/MM HH:mm'))
    } else {
        Write-Host "run completed."
    }

    if (Test-Path $Log) {
        Write-Host ""
        Write-Host "--- anomalies in the log ---"
        $suspect = @(Select-String -Path $Log -Pattern 'failed|Traceback|Error|BrokenPipe|RESTART' `
                                   -CaseSensitive -ErrorAction SilentlyContinue)
        if ($suspect.Count -eq 0) {
            Write-Host "  none (0 fallbacks from the C++ engine, 0 exceptions, 0 restarts)"
        } else {
            Write-Host "  $($suspect.Count) suspicious lines, last 5:"
            foreach ($s in ($suspect | Select-Object -Last 5)) {
                Write-Host "    $($s.Line)"
            }
        }
    }
}

if ($Follow) {
    while ($true) {
        # Clear-Host breaks when there is no real console -- output redirected to
        # a file, or running inside another tool. Clearing the screen is a nicety,
        # not the job of this script: if it cannot be done, carry on instead of
        # dying at the first round.
        try { Clear-Host } catch { Write-Host "" }
        Show
        Start-Sleep -Seconds $Every
    }
} else {
    Show
}
