# Runs the whole mini project on Windows, from setup to GitHub.
# Start it with:  powershell -ExecutionPolicy Bypass -File run_all.ps1   (or double-click run_all.bat)

$ErrorActionPreference = "Continue"   # native-command errors are checked in Run()
Set-Location $PSScriptRoot
$RepoName = "1MS25CI031-mlflow-champion-challenger"
$Trailer = "`n`nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`nClaude-Session: https://claude.ai/code/session_01GMoD1tzXsHMbPL9WKxts7e"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Run($exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $exe $($argv -join ' ')" }
}
function Commit($msg) {
    git add -A
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) { Run git @("commit", "-q", "-m", "$msg$Trailer") ; Write-Host "   committed: $msg" }
}

# ---------------------------------------------------------------- 1. git + rule first
Step "Git repository"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Git is not installed. Install it from https://git-scm.com and run this again." }
if (-not (Test-Path .git)) { Run git @("init", "-q", "-b", "main") }
if (-not (git config user.name))  { git config user.name "Bhavayraj Singhal" }
if (-not (git config user.email)) { $e = Read-Host "Email address of your GitHub account (for the commits)"; git config user.email $e }
if (-not (git rev-parse --verify HEAD 2>$null)) {
    git add README.md champion_rule.json requirements.txt .gitignore data/new_data.csv
    Run git @("commit", "-q", "-m", "Project setup and Champion rule, written before any evaluation run$Trailer")
    Write-Host "   committed: Champion rule"
    Commit "Add evaluation, registry, prediction and report scripts"
}

# ---------------------------------------------------------------- 2. python environment
Step "Python environment (.venv)"
$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
if (-not (Test-Path .venv\Scripts\python.exe)) {
    if ($py -eq "py") { Run py @("-3", "-m", "venv", ".venv") } else { Run python @("-m", "venv", ".venv") }
}
$PY = (Resolve-Path .venv\Scripts\python.exe).Path
Run $PY @("-c", "import sys; assert (3,9) <= sys.version_info[:2] <= (3,13), 'Use Python 3.9 to 3.13 (PyTorch does not support ' + sys.version.split()[0] + ')'")
Run $PY @("-m", "pip", "install", "-q", "--upgrade", "pip")
Run $PY @("-m", "pip", "install", "-q", "-r", "requirements.txt")
Run $PY @("-m", "playwright", "install", "chromium")

# ---------------------------------------------------------------- 3. MLflow server
Step "MLflow tracking server on http://127.0.0.1:5000"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
function ServerUp { try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5000/health -TimeoutSec 2).StatusCode -eq 200 } catch { $false } }
if (-not (ServerUp)) {
    Start-Process -FilePath $PY -ArgumentList "-m", "mlflow", "server", "--host", "127.0.0.1", "--port", "5000" `
        -RedirectStandardOutput mlflow_server.log -RedirectStandardError mlflow_server.err.log -WindowStyle Hidden
    for ($i = 0; $i -lt 60 -and -not (ServerUp); $i++) { Start-Sleep -Seconds 2 }
    if (-not (ServerUp)) { throw "MLflow server did not start. See mlflow_server.err.log" }
}
Write-Host "   server is up"

# ---------------------------------------------------------------- 4. experiment
Step "Evaluation data (200 SST-2 sentences, seed 42)"
if (-not (Test-Path data\eval_inputs.csv)) { Run $PY @("prepare_data.py") }
Commit "Add shared evaluation set data/eval_inputs.csv"

Step "Running both models (first run downloads ~1.7 GB of models)"
Run $PY @("evaluate.py")
Step "Registering models and setting aliases"
Run $PY @("register_champion.py")
Step "Predicting on new data with @champion"
Run $PY @("predict_from_mlflow.py")
Step "Stretch goal: same script with @challenger"
Run $PY @("predict_from_mlflow.py", "--alias", "challenger")
Step "Screenshots"
Run $PY @("capture_screenshots.py")
Step "Writing results and decision into README.md"
Run $PY @("write_report.py")
Commit "Add results, plots, screenshots, registry aliases and Champion decision"

# ---------------------------------------------------------------- 5. GitHub
Step "GitHub"
$hasRemote = git remote | Select-String -Quiet origin
if ($hasRemote) {
    Run git @("push", "-q", "origin", "main")
    Write-Host "   pushed to origin"
} elseif ((Get-Command gh -ErrorAction SilentlyContinue) -and ((gh auth status 2>&1) -match "Logged in")) {
    Run gh @("repo", "create", $RepoName, "--public", "--source", ".", "--remote", "origin", "--push")
    Write-Host "   created and pushed: https://github.com/$(gh api user --jq .login)/$RepoName"
} else {
    Write-Host "   Create an EMPTY public repo named $RepoName on https://github.com/new (no README), then run:" -ForegroundColor Yellow
    Write-Host "   git remote add origin https://github.com/<your-username>/$RepoName.git"
    Write-Host "   git push -u origin main"
}

Write-Host "`nDone. Open http://127.0.0.1:5000 to see the runs. Read README.md sections 4 and 5 before submitting." -ForegroundColor Green
