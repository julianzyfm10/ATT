param([string]$WaitForProcess = '')

$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot 'run_common_judge.py'
$runs = Join-Path $PSScriptRoot 'common_judge'
$qwen = 'Qwen/Qwen3.6-35B-A3B'
$gemma = 'google/gemma-4-31B-it'
$limits = @('--input-byte-limit', '160000', '--workers', '4', '--request-interval', '1.5')

function Invoke-Judge {
    & python -u $script @args
    if ($LASTEXITCODE -ne 0) {
        throw "Judge command exited with status $LASTEXITCODE. Resume by rerunning this script."
    }
}

$lock = [System.IO.File]::Open((Join-Path $runs 'pipeline.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
try {
    Set-Location (Resolve-Path (Join-Path $PSScriptRoot '../..'))
    foreach ($processId in ($WaitForProcess -split ',' | Where-Object { $_ })) {
        $process = Get-Process -Id ([int]$processId) -ErrorAction SilentlyContinue
        if ($process) {
            Write-Output "Waiting for active development process $processId"
            $process.WaitForExit()
        }
    }

    $mainDev = Join-Path $runs 'qwen-dev.sqlite'
    $mainTest = Join-Path $runs 'qwen-test.sqlite'
    $reviewDev = Join-Path $runs 'gemma-dev.sqlite'
    $review = Join-Path $runs 'gemma-review.sqlite'

    Write-Output 'Qwen development'
    Invoke-Judge run --split development --model $qwen --run-file $mainDev @limits
    Invoke-Judge score --split development --run-file $mainDev
    Write-Output 'Gemma development'
    Invoke-Judge run --split development --model $gemma --run-file $reviewDev @limits
    Invoke-Judge score --split development --run-file $reviewDev

    Write-Output 'Qwen test'
    Invoke-Judge run --split test --model $qwen --development-run $mainDev --run-file $mainTest @limits
    Invoke-Judge score --split test --run-file $mainTest
    Write-Output 'Gemma paired review'
    Invoke-Judge run --split review --model $gemma --development-run $reviewDev --primary-run $mainTest --run-file $review @limits
    Invoke-Judge score --split review --primary-run $mainTest --run-file $review
    Write-Output 'Exp1 common judge complete'
} finally {
    $lock.Dispose()
}
