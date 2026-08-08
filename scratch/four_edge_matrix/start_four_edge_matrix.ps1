param(
    [int]$MaxCount = 5000,
    [int]$PollSeconds = 20,
    [int]$RunTimeoutHours = 36,
    [int]$SshTimeoutSeconds = 20
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LogDir = Join-Path $ProjectRoot "log\four_edge_matrix\local_140_118_238"
$ScriptPath = Join-Path $PSScriptRoot "run_four_edge_matrix.ps1"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutPath = Join-Path $LogDir "orchestrator_$timestamp.stdout.log"
$stderrPath = Join-Path $LogDir "orchestrator_$timestamp.stderr.log"
$argumentList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $ScriptPath,
    "-MaxCount", [string]$MaxCount,
    "-PollSeconds", [string]$PollSeconds,
    "-RunTimeoutHours", [string]$RunTimeoutHours,
    "-SshTimeoutSeconds", [string]$SshTimeoutSeconds
)
$process = Start-Process -FilePath "powershell.exe" `
    -ArgumentList $argumentList `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

Write-Output "started_pid=$($process.Id)"
Write-Output "stdout_log=$stdoutPath"
Write-Output "stderr_log=$stderrPath"
