param(
    [switch]$SkipCudaTorch
)

$ErrorActionPreference = "Stop"
$LabRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvRoot = Join-Path $LabRoot ".venv"
$PythonExe = Join-Path $VenvRoot "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    python -m venv $VenvRoot
}

& $PythonExe -m pip install --upgrade pip

if (-not $SkipCudaTorch) {
    & $PythonExe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
}

& $PythonExe -m pip install -r (Join-Path $LabRoot "requirements.txt")
& $PythonExe -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
