# Parse every .ps1 in the repo with the PowerShell AST parser (no execution).
# Fails on syntax errors only; the scripts themselves need the GPU PC to run.
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '../..')
$files = Get-ChildItem -Path $root -Recurse -Filter *.ps1 | Where-Object { $_.FullName -notmatch '[\/]\.git[\/]' }
$bad = 0
foreach ($f in $files) {
    $tokens = $null; $errors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile($f.FullName, [ref]$tokens, [ref]$errors)
    foreach ($e in $errors) {
        $bad++
        Write-Host "::error file=$($f.FullName),line=$($e.Extent.StartLineNumber)::$($e.Message)"
    }
}
Write-Host "Parsed $($files.Count) PowerShell scripts, $bad error(s)"
if ($bad -gt 0) { exit 1 }
