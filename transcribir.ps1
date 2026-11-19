param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $CliArgs
)

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $venvPython) {
    & $venvPython -m transcribe_audio @CliArgs
} else {
    python -m transcribe_audio @CliArgs
}
exit $LASTEXITCODE
