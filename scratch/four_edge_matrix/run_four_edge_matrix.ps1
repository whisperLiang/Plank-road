param(
    [int]$MaxCount = 5000,
    [int]$PollSeconds = 20,
    [int]$RunTimeoutHours = 36
)

$ErrorActionPreference = "Stop"
$MatrixDir = $PSScriptRoot
$ProjectRoot = (Resolve-Path (Join-Path $MatrixDir "..\..")).Path
$DeviceSetSlug = "local_140_118_238"
$StateDir = Join-Path $MatrixDir "state_$DeviceSetSlug"
$RecordDir = Join-Path $StateDir "records"
New-Item -ItemType Directory -Force -Path $StateDir, $RecordDir | Out-Null
Set-Content -LiteralPath (Join-Path $StateDir "orchestrator.pid") -Value $PID -Encoding ascii

$Cloud = [ordered]@{
    Host = "whisperliang@192.168.66.205"
    Address = "192.168.66.205"
    Project = "/home/whisperliang/RECAP"
    Python = ".venv/bin/python"
}

$Edges = @(
    [ordered]@{ Id = 1; Host = $null; Project = $ProjectRoot; Python = (Join-Path $ProjectRoot ".venv\Scripts\python.exe") },
    [ordered]@{ Id = 2; Host = "nvidia@192.168.66.140"; Project = "/home/nvidia/RECAP"; Python = ".venv/bin/python" },
    [ordered]@{ Id = 3; Host = "nvidia@192.168.66.118"; Project = "/home/nvidia/RECAP"; Python = ".venv/bin/python" },
    [ordered]@{ Id = 4; Host = "nvidia@192.168.66.238"; Project = "/home/nvidia/RECAP"; Python = ".venv/bin/python" }
)

$Models = @(
    [ordered]@{ Name = "yolo26n"; Config = "scratch/four_edge_matrix/config_yolo26n.yaml"; Experiment = "weather_model_comparison_yolo26n_$DeviceSetSlug" },
    [ordered]@{ Name = "rfdetr_nano"; Config = "scratch/four_edge_matrix/config_rfdetr_nano.yaml"; Experiment = "weather_model_comparison_rfdetr_nano_$DeviceSetSlug" }
)
$Scenarios = @("rainy", "snowy")
$Methods = @("recap", "SURGEON", "CATR", "Ekya")
$SshOptions = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3")

function Assert-SafeToken {
    param([string]$Value)
    if ($Value -notmatch '^[A-Za-z0-9_./:=@-]+$') {
        throw "Unsafe shell token: $Value"
    }
    return $Value
}

function Invoke-Remote {
    param(
        [string]$HostName,
        [string]$Command
    )
    $output = & ssh @SshOptions $HostName $Command 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "SSH command failed on $HostName (exit $LASTEXITCODE): $output"
    }
    return (($output | Out-String).Trim())
}

function Start-RemoteJob {
    param(
        [System.Collections.IDictionary]$Node,
        [string]$Name,
        [string]$Entrypoint,
        [string[]]$Arguments
    )
    $tokens = @($Node.Python, $Entrypoint) + $Arguments
    $tokens = @($tokens | ForEach-Object { Assert-SafeToken ([string]$_) })
    $command = "cd $($Node.Project) && bash scratch/four_edge_matrix/remote_job.sh start $Name " + ($tokens -join " ")
    Invoke-Remote -HostName $Node.Host -Command $command | Out-Null
}

function Get-RemoteJobStatus {
    param(
        [System.Collections.IDictionary]$Node,
        [string]$Name
    )
    try {
        return Invoke-Remote -HostName $Node.Host -Command "cd $($Node.Project) && bash scratch/four_edge_matrix/remote_job.sh status $Name"
    }
    catch {
        return "SSH_ERROR"
    }
}

function Stop-RemoteJob {
    param(
        [System.Collections.IDictionary]$Node,
        [string]$Name
    )
    try {
        Invoke-Remote -HostName $Node.Host -Command "cd $($Node.Project) && bash scratch/four_edge_matrix/remote_job.sh stop $Name" | Out-Null
    }
    catch {
        Write-Warning $_
    }
}

function Test-TcpPort {
    param(
        [string]$Address,
        [int]$Port = 50051,
        [int]$TimeoutMilliseconds = 1500
    )
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync($Address, $Port)
        return $task.Wait($TimeoutMilliseconds) -and $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Wait-CloudReady {
    param([string]$CloudJob)
    $deadline = [DateTime]::UtcNow.AddMinutes(5)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-TcpPort -Address $Cloud.Address) {
            return
        }
        $status = Get-RemoteJobStatus -Node $Cloud -Name $CloudJob
        if ($status -match '^\d+$') {
            throw "Cloud exited before opening port 50051 (exit $status)."
        }
        Start-Sleep -Seconds 3
    }
    throw "Cloud did not open port 50051 within five minutes."
}

function Wait-CloudStopped {
    $deadline = [DateTime]::UtcNow.AddMinutes(2)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (-not (Test-TcpPort -Address $Cloud.Address)) {
            return
        }
        Start-Sleep -Seconds 2
    }
    throw "Cloud port 50051 remained open after stop."
}

function Write-MatrixStatus {
    param([System.Collections.IDictionary]$Status)
    $Status["updated_at"] = [DateTime]::Now.ToString("o")
    $Status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $StateDir "matrix_status.json") -Encoding utf8
}

foreach ($edge in $Edges) {
    if ($null -eq $edge.Host) {
        if (-not (Test-Path -LiteralPath $edge.Python)) {
            throw "Local Python is missing: $($edge.Python)"
        }
        continue
    }
    Invoke-Remote -HostName $edge.Host -Command "cd $($edge.Project) && test -f scratch/four_edge_matrix/remote_job.sh && test -f scratch/four_edge_matrix/config_yolo26n.yaml && test -f scratch/four_edge_matrix/config_rfdetr_nano.yaml" | Out-Null
}
Invoke-Remote -HostName $Cloud.Host -Command "cd $($Cloud.Project) && test -f scratch/four_edge_matrix/remote_job.sh && test -f scratch/four_edge_matrix/config_yolo26n.yaml && test -f scratch/four_edge_matrix/config_rfdetr_nano.yaml" | Out-Null

if (Test-TcpPort -Address $Cloud.Address) {
    throw "Cloud port 50051 is already in use; refusing to stop an unowned process."
}

$matrixStarted = [DateTime]::Now
$successfulRuns = 0
$failedRuns = 0

foreach ($model in $Models) {
    foreach ($scenario in $Scenarios) {
        foreach ($method in $Methods) {
            $methodSlug = $method.ToLowerInvariant()
            $runName = "$($model.Name)_${scenario}_n4_r01_${methodSlug}_set238"
            $recordPath = Join-Path $RecordDir "$runName.json"
            if (Test-Path -LiteralPath $recordPath) {
                $existing = Get-Content -Raw -LiteralPath $recordPath | ConvertFrom-Json
                if ($existing.status -eq "success") {
                    $successfulRuns++
                    continue
                }
            }

            $cloudJob = "$runName-cloud"
            $runStarted = [DateTime]::Now
            $runStatus = "running"
            $failureReason = $null
            $localProcess = $null
            $remoteStatuses = [ordered]@{ edge2 = "NOT_STARTED"; edge3 = "NOT_STARTED"; edge4 = "NOT_STARTED" }

            $commonCloudArgs = @(
                "--yaml_path", $model.Config,
                "--experiment_id", $model.Experiment,
                "--scenario", $scenario,
                "--edge_count", "4",
                "--repeat", "1",
                "--experiment_results_root", "results/experiments",
                "--workspace_root", "./cache/server_workspace/n4/$runName"
            )
            if ($method -eq "recap") {
                $cloudArgs = $commonCloudArgs + @("--mode", "main")
            }
            else {
                $cloudArgs = $commonCloudArgs + @("--mode", "baseline", "--baseline_method", $method)
            }

            try {
                if (Test-TcpPort -Address $Cloud.Address) {
                    throw "Cloud port 50051 is unexpectedly busy before $runName."
                }
                Start-RemoteJob -Node $Cloud -Name $cloudJob -Entrypoint "cloud_server.py" -Arguments $cloudArgs
                Wait-CloudReady -CloudJob $cloudJob

                foreach ($edge in $Edges | Where-Object { $null -ne $_.Host }) {
                    $edgeJob = "$runName-edge$($edge.Id)"
                    $edgeArgs = @(
                        "--yaml_path", $model.Config,
                        "--edge_id", [string]$edge.Id,
                        "--cache_path", "./cache/n4/$runName/edge_$($edge.Id)",
                        "--video_path", "./video_data/$scenario.mp4",
                        "--server_ip", "192.168.66.205:50051",
                        "--max_count", [string]$MaxCount,
                        "--headless",
                        "--experiment_id", $model.Experiment,
                        "--scenario", $scenario,
                        "--edge_count", "4",
                        "--repeat", "1",
                        "--experiment_results_root", "./cache/experiment_results"
                    )
                    if ($method -eq "recap") {
                        $edgeArgs += @("--mode", "main")
                    }
                    else {
                        $edgeArgs += @("--mode", "baseline", "--baseline_method", $method)
                    }
                    Start-RemoteJob -Node $edge -Name $edgeJob -Entrypoint "edge_client.py" -Arguments $edgeArgs
                }

                $localStdout = Join-Path $StateDir "$runName-edge1.stdout.log"
                $localStderr = Join-Path $StateDir "$runName-edge1.stderr.log"
                $localArgs = @(
                    "edge_client.py",
                    "--yaml_path", $model.Config,
                    "--edge_id", "1",
                    "--cache_path", "./cache/n4/$runName/edge_1",
                    "--video_path", "./video_data/$scenario.mp4",
                    "--server_ip", "192.168.66.205:50051",
                    "--max_count", [string]$MaxCount,
                    "--headless",
                    "--experiment_id", $model.Experiment,
                    "--scenario", $scenario,
                    "--edge_count", "4",
                    "--repeat", "1",
                    "--experiment_results_root", "./cache/experiment_results"
                )
                if ($method -eq "recap") {
                    $localArgs += @("--mode", "main")
                }
                else {
                    $localArgs += @("--mode", "baseline", "--baseline_method", $method)
                }
                $localProcess = Start-Process -FilePath $Edges[0].Python -ArgumentList $localArgs -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $localStdout -RedirectStandardError $localStderr -PassThru
                Set-Content -LiteralPath (Join-Path $StateDir "$runName-edge1.pid") -Value $localProcess.Id -Encoding ascii

                $deadline = [DateTime]::UtcNow.AddHours($RunTimeoutHours)
                $startupDeadline = [DateTime]::UtcNow.AddMinutes(2)
                while ($true) {
                    $localProcess.Refresh()
                    if (-not $localProcess.HasExited) {
                        $localStatus = "RUNNING"
                    }
                    else {
                        $localExitCode = $localProcess.ExitCode
                        if ($null -ne $localExitCode -and [string]$localExitCode -ne "") {
                            $localStatus = [string]$localExitCode
                        }
                        elseif (
                            (Test-Path -LiteralPath $localStderr) -and
                            (Select-String -LiteralPath $localStderr -Pattern "streaming complete|Uploaded .*offline experiment artifact" -Quiet)
                        ) {
                            $localStatus = "0"
                        }
                        else {
                            $localStatus = "UNKNOWN"
                        }
                    }
                    foreach ($edge in $Edges | Where-Object { $null -ne $_.Host }) {
                        $remoteStatuses["edge$($edge.Id)"] = Get-RemoteJobStatus -Node $edge -Name "$runName-edge$($edge.Id)"
                    }
                    $cloudStatus = Get-RemoteJobStatus -Node $Cloud -Name $cloudJob

                    Write-MatrixStatus -Status ([ordered]@{
                        matrix_started_at = $matrixStarted.ToString("o")
                        current_run = $runName
                        model = $model.Name
                        scenario = $scenario
                        method = $method
                        max_count = $MaxCount
                        run_started_at = $runStarted.ToString("o")
                        cloud = $cloudStatus
                        edges = [ordered]@{ edge1 = $localStatus; edge2 = $remoteStatuses.edge2; edge3 = $remoteStatuses.edge3; edge4 = $remoteStatuses.edge4 }
                        successful_runs = $successfulRuns
                        failed_runs = $failedRuns
                        total_runs = 16
                    })

                    $numericRemote = @($remoteStatuses.Values | Where-Object { $_ -match '^\d+$' })
                    $remoteFailure = @($numericRemote | Where-Object { [int]$_ -ne 0 })
                    if ($localStatus -match '^\d+$' -and [int]$localStatus -ne 0) {
                        throw "Local edge exited with code $localStatus."
                    }
                    if ($remoteFailure.Count -gt 0) {
                        throw "A remote edge failed: $($remoteStatuses | ConvertTo-Json -Compress)."
                    }
                    if ($cloudStatus -match '^\d+$') {
                        throw "Cloud exited while edges were running (exit $cloudStatus)."
                    }
                    if ([DateTime]::UtcNow -gt $startupDeadline -and $remoteStatuses.Values -contains "UNKNOWN") {
                        throw "A remote edge never reached a running state: $($remoteStatuses | ConvertTo-Json -Compress)."
                    }
                    if ([DateTime]::UtcNow -gt $deadline) {
                        throw "Run exceeded the ${RunTimeoutHours}-hour timeout."
                    }

                    $allRemoteDone = @($remoteStatuses.Values | Where-Object { $_ -ne "0" }).Count -eq 0
                    if ($localStatus -eq "0" -and $allRemoteDone) {
                        break
                    }
                    Start-Sleep -Seconds $PollSeconds
                }

                $runStatus = "success"
                $successfulRuns++
            }
            catch {
                $runStatus = "failed"
                $failureReason = $_.Exception.Message
                $failedRuns++
                if ($null -ne $localProcess) {
                    $localProcess.Refresh()
                    if (-not $localProcess.HasExited) {
                        Stop-Process -Id $localProcess.Id -Force -ErrorAction SilentlyContinue
                    }
                }
                foreach ($edge in $Edges | Where-Object { $null -ne $_.Host }) {
                    Stop-RemoteJob -Node $edge -Name "$runName-edge$($edge.Id)"
                }
            }
            finally {
                Stop-RemoteJob -Node $Cloud -Name $cloudJob
                try { Wait-CloudStopped } catch { if ($null -eq $failureReason) { $failureReason = $_.Exception.Message; $runStatus = "failed" } }

                $finalLocalStatus = "NOT_STARTED"
                if ($null -ne $localProcess) {
                    $localProcess.Refresh()
                    $finalLocalStatus = if ($localProcess.HasExited) { [string]$localProcess.ExitCode } else { "STOPPED" }
                }
                foreach ($edge in $Edges | Where-Object { $null -ne $_.Host }) {
                    $remoteStatuses["edge$($edge.Id)"] = Get-RemoteJobStatus -Node $edge -Name "$runName-edge$($edge.Id)"
                }
                $record = [ordered]@{
                    run_name = $runName
                    status = $runStatus
                    failure_reason = $failureReason
                    model = $model.Name
                    scenario = $scenario
                    method = $method
                    edge_count = 4
                    repeat = 1
                    max_count = $MaxCount
                    started_at = $runStarted.ToString("o")
                    finished_at = [DateTime]::Now.ToString("o")
                    edge_exit_codes = [ordered]@{ edge1 = $finalLocalStatus; edge2 = $remoteStatuses.edge2; edge3 = $remoteStatuses.edge3; edge4 = $remoteStatuses.edge4 }
                    cloud_status = Get-RemoteJobStatus -Node $Cloud -Name $cloudJob
                }
                $record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $recordPath -Encoding utf8
                Write-MatrixStatus -Status ([ordered]@{
                    matrix_started_at = $matrixStarted.ToString("o")
                    last_run = $runName
                    last_run_status = $runStatus
                    last_failure_reason = $failureReason
                    successful_runs = $successfulRuns
                    failed_runs = $failedRuns
                    total_runs = 16
                })
                Start-Sleep -Seconds 5
            }
        }
    }
}

Write-MatrixStatus -Status ([ordered]@{
    matrix_started_at = $matrixStarted.ToString("o")
    finished_at = [DateTime]::Now.ToString("o")
    status = if ($failedRuns -eq 0) { "success" } else { "completed_with_failures" }
    successful_runs = $successfulRuns
    failed_runs = $failedRuns
    total_runs = 16
})
