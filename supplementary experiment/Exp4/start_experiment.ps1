$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$python = Join-Path $root '.venv_mcp_live/Scripts/python.exe'
$output = Join-Path $PSScriptRoot 'results'
$null = New-Item -ItemType Directory -Path $output -Force
$lock = [IO.File]::Open((Join-Path $output 'pipeline.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
try {
    Set-Location $root
    & $python -u (Join-Path $PSScriptRoot 'run_experiment.py') --output $output
    if ($LASTEXITCODE -ne 0) { throw "Exp4 exited with status $LASTEXITCODE. Completed episodes remain resumable." }
} finally {
    $lock.Dispose()
}
