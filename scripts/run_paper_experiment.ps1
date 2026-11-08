param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 4)]
    [int]$Experiment,
    [ValidateSet("deepseek", "siliconflow")]
    [string]$Provider = "deepseek",
    [string]$Model,
    [string]$PythonExe = "python",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArguments
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot ("run_experiment{0}_llm_{1}.py" -f $Experiment, $Provider)
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Experiment entry point not found: $runner"
}
$arguments = @($runner)
if ($Model) {
    $arguments += @("--model", $Model)
}
if ($RunnerArguments) {
    $arguments += $RunnerArguments
}
Push-Location -LiteralPath $root
try {
    & $PythonExe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment $Experiment failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
