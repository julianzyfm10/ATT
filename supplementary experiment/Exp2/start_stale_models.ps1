$ErrorActionPreference = 'Stop'
$output = Join-Path $PSScriptRoot 'model_results'
$null = New-Item -ItemType Directory -Path $output -Force
$lock = [IO.File]::Open((Join-Path $output 'pipeline.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
try {
    & python -u (Join-Path $PSScriptRoot 'run_stale_models.py') --output $output
    if ($LASTEXITCODE -ne 0) { throw "Exp2 stopped with exit code $LASTEXITCODE. Saved cells can be resumed." }
} finally {
    $lock.Dispose()
}
