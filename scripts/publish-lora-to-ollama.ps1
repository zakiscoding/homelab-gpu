# Publish a succeeded train job to Ollama (merge QLoRA adapter → ollama create).
# Usage:
#   .\scripts\publish-lora-to-ollama.ps1 -JobId <uuid>
#   .\scripts\publish-lora-to-ollama.ps1 -JobId <uuid> -OllamaName "homelab-smoke-lora-v1"
param(
    [Parameter(Mandatory = $true)][string]$JobId,
    [string]$OllamaName = "",
    [switch]$SkipMerge,
    [switch]$NoRestartOllama
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$venvPy = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "Run .\scripts\setup-venv.ps1 first" }

$publishPy = Join-Path $Repo "training\lora_quickstart\publish.py"
$pyArgs = @($publishPy, $JobId)
if ($OllamaName) { $pyArgs += @("--name", $OllamaName) }
if ($SkipMerge) { $pyArgs += "--skip-merge" }
if ($NoRestartOllama) { $pyArgs += "--no-restart-ollama" }

Push-Location $Repo
try {
    & $venvPy @pyArgs
} finally {
    Pop-Location
}
