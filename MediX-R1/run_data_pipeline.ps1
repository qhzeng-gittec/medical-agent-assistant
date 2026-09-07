$ErrorActionPreference = "Stop"
$LabRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $LabRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Missing .venv. Run setup_env.ps1 first."
}

Push-Location $LabRoot
try {
    & $PythonExe prepare_vqa_project_split.py
    if ($LASTEXITCODE -ne 0) { throw "Original image preparation failed." }
    & $PythonExe complete_reasoning.py
    if ($LASTEXITCODE -ne 0) { throw "Reasoning annotation failed." }
    & $PythonExe prepare_open_cases.py
    if ($LASTEXITCODE -ne 0) { throw "Open-question conversion failed." }
    & $PythonExe build_native_recipes.py
    if ($LASTEXITCODE -ne 0) { throw "Native recipe preparation failed." }
    & $PythonExe -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Native data and supervision checks failed." }
    & $PythonExe train_multitask_lora.py --recipe all_reasoning --dry-run
    if ($LASTEXITCODE -ne 0) { throw "Training dry run failed." }
}
finally {
    Pop-Location
}
