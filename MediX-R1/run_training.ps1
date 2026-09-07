param(
    [string]$OutputRoot = "outputs\reasoning_experiments",
    [string]$EvaluationRoot = "outputs\reasoning_evaluations",
    [string]$JudgeModel = "gpt-5.6-sol"
)
$ErrorActionPreference = "Stop"
$LabRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $LabRoot ".venv\Scripts\python.exe"
Push-Location $LabRoot
try {
    $name = "all_reasoning_attention_ffn"
    $output = Join-Path $OutputRoot $name
    if (Test-Path -LiteralPath "$output\final_adapter") { throw "Adapter already exists: $output" }
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    & $PythonExe train_multitask_lora.py --recipe all_reasoning --lora-scope attention_ffn --output-root $OutputRoot
    if ($LASTEXITCODE -ne 0) { throw "Training failed." }
    & $PythonExe evaluate_native_reasoning.py --name $name --adapter-dir "$output\final_adapter" --output-root $EvaluationRoot --judge-model $JudgeModel --split test --limit-per-task 20
    if ($LASTEXITCODE -ne 0) { throw "LLM judge evaluation failed." }
}
finally { Pop-Location }
