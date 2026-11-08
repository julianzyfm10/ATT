[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [switch]$RunConnectionTest,
    [switch]$RunMcpPreflight,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Launcher = Join-Path $PSScriptRoot "start_siliconflow_full.ps1"
$Dataset = Join-Path $Root "data\benchmark\experiment4_mcpbench_core_local_30.jsonl"
$Manifest = Join-Path $Root "data\benchmark\experiment4_mcpbench_core_local_30_manifest.json"

if (-not (Test-Path -LiteralPath $Launcher)) {
    throw "SiliconFlow launcher not found: $Launcher"
}
if (-not (Test-Path -LiteralPath $Dataset)) {
    throw "Experiment 4 Core Local dataset not found: $Dataset"
}
if (-not (Test-Path -LiteralPath $Manifest)) {
    throw "Experiment 4 Core Local manifest not found: $Manifest"
}

$datasetRows = @(Get-Content -LiteralPath $Dataset | Where-Object { $_.Trim() })
$manifestData = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
if ($datasetRows.Count -ne 30 -or [int]$manifestData.case_count -ne 30) {
    throw "Expected the frozen 30-case Core Local dataset; JSONL=$($datasetRows.Count), manifest=$($manifestData.case_count)."
}

if (-not $PythonExe) {
    $bundled = Join-Path $Root ".venv_mcp_live\Scripts\python.exe"
    $PythonExe = if (Test-Path -LiteralPath $bundled) { $bundled } else { "python" }
}

if ($RunMcpPreflight) {
    Write-Host "Checking the five Experiment 4 local MCP servers once..." -ForegroundColor Cyan
    & $PythonExe (Join-Path $PSScriptRoot "preflight_experiment4_core_local.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment 4 MCP preflight failed. No model process was started."
    }
}

$jobs = @(
    [pscustomobject]@{
        Name = "GPT-OSS"
        Model = "gpt-oss-120b"
        FullModel = "openai/gpt-oss-120b"
        Slug = "gpt_oss_120b"
        DisableThinking = $false
    },
    [pscustomobject]@{
        Name = "Gemma"
        Model = "gemma-4-31b-it"
        FullModel = "google/gemma-4-31B-it"
        Slug = "gemma_4_31b_it"
        DisableThinking = $false
    },
    [pscustomobject]@{
        Name = "Qwen"
        Model = "qwen3.6-35b-a3b"
        FullModel = "Qwen/Qwen3.6-35B-A3B"
        Slug = "qwen3_6_35b_a3b"
        DisableThinking = $true
    }
)

$PowerShellExe = Join-Path $PSHOME "powershell.exe"
$started = [System.Collections.Generic.List[object]]::new()

foreach ($job in $jobs) {
    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $Launcher,
        "-Experiment", "4",
        "-Model", $job.Model,
        "-Background",
        "-SkipExperiment4Preflight"
    )
    if ($PythonExe) {
        $arguments += @("-PythonExe", $PythonExe)
    }
    if (-not $RunConnectionTest) {
        $arguments += "-SkipConnectionTest"
    }
    if ($NoResume) {
        $arguments += "-NoResume"
    }
    if ($job.DisableThinking) {
        # This is inherited only by the Qwen worker and becomes
        # extra_body={"enable_thinking": false} on every API call.
        $arguments += "-DisableThinking"
    }

    Write-Host "Starting Experiment 4 in background: $($job.Name)" -ForegroundColor Cyan
    $launcherOutput = & $PowerShellExe @arguments
    $exitCode = $LASTEXITCODE
    $launcherOutput | ForEach-Object { Write-Host $_ }
    if ($exitCode -ne 0) {
        throw "$($job.Name) failed to start; launcher exit code: $exitCode"
    }

    $pidPath = Join-Path $Root "outputs\logs\experiment4_llm_siliconflow_$($job.Slug)_full_pid.txt"
    $pidValue = if (Test-Path -LiteralPath $pidPath) {
        (Get-Content -LiteralPath $pidPath -Raw).Trim()
    } else {
        "unknown"
    }
    $started.Add([pscustomobject]@{
        Model = $job.FullModel
        PID = $pidValue
        Thinking = if ($job.DisableThinking) { "disabled" } else { "provider default" }
        Trace = "outputs/core_local/experiment4_core_local_llm_$($job.Slug)_full_traces.jsonl"
    })
}

Write-Host ""
Write-Host "All three Experiment 4 model processes were submitted in background." -ForegroundColor Green
Write-Host "Protocol per model: 30 cases x 4 strategies x 3 backends = 360 runs."
Write-Host "Dataset and metrics: identical to the completed DeepSeek Core Local experiment."
Write-Host "Qwen thinking: explicitly disabled; GPT-OSS/Gemma use provider defaults."
$started | Format-Table -AutoSize

Write-Host "Progress files:" -ForegroundColor Cyan
foreach ($job in $jobs) {
    Write-Host "  outputs\core_local\experiment4_core_local_llm_$($job.Slug)_full_summary.json"
}
