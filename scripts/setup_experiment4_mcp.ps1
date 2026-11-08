param([string]$PythonExe = "python")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$vendor = Join-Path $root "external/mcp-bench"
$revision = "7a8eaeae83a842a2949080acc5473f65e1569daf"

if (-not (Test-Path -LiteralPath $vendor)) {
    git clone https://github.com/Accenture/mcp-bench.git $vendor
    if ($LASTEXITCODE -ne 0) { throw "MCP-Bench clone failed" }
    git -C $vendor checkout --detach $revision
    if ($LASTEXITCODE -ne 0) { throw "MCP-Bench revision checkout failed" }
}
$actual = (git -C $vendor rev-parse HEAD).Trim()
if ($actual -ne $revision) {
    throw "MCP-Bench revision $actual differs from the tested revision $revision. Check out that revision before setup."
}

$environment = Join-Path $root ".venv_mcp_live"
if (-not (Test-Path -LiteralPath $environment)) {
    & $PythonExe -m venv $environment
    if ($LASTEXITCODE -ne 0) { throw "Python virtual environment creation failed" }
}
$python = Join-Path $environment "Scripts/python.exe"
& $python -m pip install --upgrade pip mcp fastmcp numpy scipy sympy matplotlib `
    -e (Join-Path $vendor "mcp_servers/unit-converter-mcp") `
    -e (Join-Path $vendor "mcp_servers/time-mcp") `
    -e (Join-Path $vendor "mcp_servers/medcalc")
if ($LASTEXITCODE -ne 0) { throw "Python MCP dependency installation failed" }

$math = Join-Path $vendor "mcp_servers/math-mcp"
Push-Location -LiteralPath $math
try {
    npm ci
    if ($LASTEXITCODE -ne 0) { throw "Math MCP npm install failed" }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "Math MCP build failed" }
}
finally {
    Pop-Location
}
Write-Host "Experiment 4 MCP servers are installed. Run scripts/preflight_experiment4_core_local.py next."
