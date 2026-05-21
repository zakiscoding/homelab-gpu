# Push RTX 5070 metrics to iMac Prometheus Pushgateway (Phase 5).
$ErrorActionPreference = "Stop"
$PushUrl = $env:PUSHGATEWAY_URL
if (-not $PushUrl) { $PushUrl = "http://100.91.113.98:9091" }
$Instance = $env:GPU_INSTANCE
if (-not $Instance) { $Instance = $env:COMPUTERNAME }
if (-not $Instance) { $Instance = "zak_pc" }

$smi = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu `
    --format=csv,noheader,nounits 2>$null
if (-not $smi) {
    Write-Error "nvidia-smi failed"
}
$parts = ($smi -split "," | ForEach-Object { $_.Trim() })
$util = [double]$parts[0]
$memUsed = [double]$parts[1]
$memTotal = [double]$parts[2]
$temp = [double]$parts[3]

$body = @"
# TYPE homelab_gpu_utilization_percent gauge
homelab_gpu_utilization_percent{instance="$Instance",job="gpu"} $util
# TYPE homelab_gpu_memory_used_mib gauge
homelab_gpu_memory_used_mib{instance="$Instance",job="gpu"} $memUsed
# TYPE homelab_gpu_memory_total_mib gauge
homelab_gpu_memory_total_mib{instance="$Instance",job="gpu"} $memTotal
# TYPE homelab_gpu_temperature_c gauge
homelab_gpu_temperature_c{instance="$Instance",job="gpu"} $temp
"@

$uri = "$PushUrl/metrics/job/gpu/instance/$Instance"
Invoke-RestMethod -Uri $uri -Method Put -Body $body -ContentType "text/plain; version=0.0.4"
Write-Output "pushed gpu metrics instance=$Instance util=$util% mem=${memUsed}/${memTotal}MiB"
