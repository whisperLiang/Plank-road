param(
    [Parameter(Mandatory = $true)][string]$PythonPath,
    [Parameter(Mandatory = $true)][string]$WorkingDirectory,
    [Parameter(Mandatory = $true)][string]$StdoutPath,
    [Parameter(Mandatory = $true)][string]$StderrPath,
    [Parameter(Mandatory = $true)][string]$ExitFile,
    [Parameter(Mandatory = $true)][string]$ArgumentsString
)

$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $WorkingDirectory
$ArgumentList = @($ArgumentsString -split ',')
& $PythonPath @ArgumentList 1> $StdoutPath 2> $StderrPath
$exitCode = if ($null -eq $LASTEXITCODE) { 1 } else { [int]$LASTEXITCODE }
Set-Content -LiteralPath $ExitFile -Value ([string]$exitCode) -Encoding ascii
exit $exitCode
