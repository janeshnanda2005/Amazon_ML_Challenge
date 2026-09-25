# ============================================================================
# run_pipeline.ps1
# Business Entity Resolution Pipeline - PowerShell Automation Launcher
# ============================================================================
#
# Usage:
#   .\run_pipeline.ps1                  # train on real data already in dataset/
#   .\run_pipeline.ps1 -Synthetic       # generate synthetic data then run
#   .\run_pipeline.ps1 -SkipTrain       # predict only (model must exist)
#   .\run_pipeline.ps1 -SkipValidate    # skip final validator check
#   .\run_pipeline.ps1 -CheckIds        # also verify matched IDs exist in test set
#   .\run_pipeline.ps1 -LightGBM        # use LightGBM classifier backend
#   .\run_pipeline.ps1 -RapidFuzz       # use rapidfuzz string metrics
#   .\run_pipeline.ps1 -Install         # pip install -r requirements.txt first
#   .\run_pipeline.ps1 -Synthetic -Install   # full first-run from scratch
#
# Run from the submission_package\ directory (one level above code\).
# ============================================================================
[CmdletBinding()]
param(
    [switch]$Synthetic,
    [switch]$SkipTrain,
    [switch]$SkipValidate,
    [switch]$CheckIds,
    [switch]$LightGBM,
    [switch]$RapidFuzz,
    [switch]$Benchmark,
    [int]$BenchmarkRows = 10000,
    [switch]$ConvertParquet,
    [switch]$Resume,
    [switch]$Install,
    [string]$PythonExe = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── colours ──────────────────────────────────────────────────────────────────
function Write-Cyan  ($msg) { Write-Host $msg -ForegroundColor Cyan }
function Write-Green ($msg) { Write-Host $msg -ForegroundColor Green }
function Write-Yellow($msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Red   ($msg) { Write-Host $msg -ForegroundColor Red }

function Write-Banner($title, $sub = "") {
    $bar = "=" * 72
    Write-Cyan ""
    Write-Cyan $bar
    Write-Cyan "  $title"
    if ($sub) { Write-Cyan "  $sub" }
    Write-Cyan $bar
    Write-Cyan ""
}

function Write-Step($title) {
    $bar = "-" * 72
    Write-Cyan ""
    Write-Cyan $bar
    Write-Cyan "  $title"
    Write-Cyan $bar
}

function Write-OK   ($msg) { Write-Green   "  [OK]   $msg" }
function Write-Warn ($msg) { Write-Yellow  "  [WARN] $msg" }
function Write-Fail ($msg) { Write-Red     "  [FAIL] $msg" }
function Write-Info ($msg) { Write-Host    "  [INFO] $msg" }

# ── resolve paths ─────────────────────────────────────────────────────────────
$ScriptDir  = $PSScriptRoot                          # submission_package\
if (Test-Path (Join-Path $ScriptDir "code\business_entity_resolution\run_pipeline.py")) {
    $CodeDir = Join-Path $ScriptDir "code\business_entity_resolution"
} else {
    $CodeDir = Join-Path $ScriptDir "code"
}
$OrchestratorPy = Join-Path $CodeDir "run_pipeline.py"
$RequirementsFile = Join-Path $CodeDir "requirements.txt"
$SyntheticPy = Join-Path $CodeDir "make_synthetic_data.py"

# ── resolve python executable ──────────────────────────────────────────────────
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $VenvPy = Join-Path $ScriptDir ".venv\Scripts\python.exe"
    if (Test-Path $VenvPy) {
        $PythonExe = $VenvPy
    } else {
        $PythonExe = "python"
    }
}

# ── banner ────────────────────────────────────────────────────────────────────
$StartTime = Get-Date
Write-Banner `
    "Business Entity Resolution Pipeline" `
    "PowerShell Launcher  |  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

Write-Info "Script dir  : $ScriptDir"
Write-Info "Code dir    : $CodeDir"
Write-Info "Python exe  : $PythonExe"
Write-Info "Flags       : Benchmark=$Benchmark  ConvertParquet=$ConvertParquet  Resume=$Resume  LightGBM=$LightGBM  RapidFuzz=$RapidFuzz"

# ── verify Python ─────────────────────────────────────────────────────────────
Write-Step "Verifying Python installation"
try {
    $pyVer = & $PythonExe --version 2>&1
    Write-OK "Found: $pyVer"
} catch {
    Write-Fail "Python executable '$PythonExe' not found. Install Python 3.9+ and try again."
    exit 1
}

# ── optional pip install ───────────────────────────────────────────────────────
if ($Install) {
    Write-Step "Installing requirements"
    Write-Info "Running: pip install -r $RequirementsFile"
    & $PythonExe -m pip install -r $RequirementsFile
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "pip install failed. Check your internet connection / proxy settings."
        exit 1
    }

    if ($LightGBM) {
        Write-Info "Installing lightgbm..."
        & $PythonExe -m pip install lightgbm
    }
    if ($RapidFuzz) {
        Write-Info "Installing rapidfuzz..."
        & $PythonExe -m pip install rapidfuzz
    }
    Write-OK "Requirements installed"
}

# ── verify orchestrator exists ────────────────────────────────────────────────
if (-not (Test-Path $OrchestratorPy)) {
    Write-Fail "run_pipeline.py not found at $OrchestratorPy"
    exit 1
}

# ── build argument list for run_pipeline.py ───────────────────────────────────
$PyArgs = @()
if ($Synthetic)      { $PyArgs += "--synthetic" }
if ($SkipTrain)      { $PyArgs += "--skip-train" }
if ($SkipValidate)   { $PyArgs += "--skip-validate" }
if ($CheckIds)       { $PyArgs += "--check-ids" }
if ($LightGBM)       { $PyArgs += "--lightgbm" }
if ($RapidFuzz)      { $PyArgs += "--rapidfuzz" }
if ($Benchmark)      { $PyArgs += "--benchmark"; $PyArgs += "--benchmark-rows"; $PyArgs += "$BenchmarkRows" }
if ($ConvertParquet) { $PyArgs += "--convert-parquet" }
if ($Resume)         { $PyArgs += "--resume" }

# ── run the Python orchestrator ───────────────────────────────────────────────
Write-Step "Launching Python pipeline orchestrator"
Write-Info "Command: $PythonExe run_pipeline.py $($PyArgs -join ' ')"
Write-Cyan ""

Push-Location $CodeDir
try {
    & $PythonExe run_pipeline.py @PyArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

# ── final report ──────────────────────────────────────────────────────────────
$Duration = (Get-Date) - $StartTime
$DurStr   = "{0:mm\:ss}" -f [timespan]$Duration

Write-Cyan ""
if ($exitCode -eq 0) {
    Write-Banner `
        "SUCCESS  (total time: $DurStr)" `
        "Outputs in:  code\business_entity_resolution\output\"
    Write-OK "matching_results.tsv  ->  upload this to the leaderboard"
    Write-OK "candidate_pairs.tsv   ->  include in your submission zip"
    Write-Cyan ""
} else {
    Write-Banner "PIPELINE FAILED  (exit code $exitCode)"
    Write-Fail "Check the output above for the stage that failed."
    Write-Fail "Common fixes:"
    Write-Info "  - Missing data  : run with -Synthetic flag for synthetic data"
    Write-Info "  - Missing deps  : run with -Install flag to pip install"
    Write-Info "  - No model      : remove -SkipTrain to retrain"
    Write-Cyan ""
    exit $exitCode
}
