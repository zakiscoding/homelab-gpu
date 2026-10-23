# Push RTX 5070 + host CPU/RAM + service health to iMac Pushgateway (Phase 5).
$ErrorActionPreference = "Stop"
$PushUrl = $env:PUSHGATEWAY_URL
if (-not $PushUrl) { $PushUrl = "http://100.91.113.98:9091" }
$Instance = $env:GPU_INSTANCE
if (-not $Instance) { $Instance = $env:COMPUTERNAME }
if (-not $Instance) { $Instance = "zak_pc" }

$smi = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu `
    --format=csv,noheader,nounits 2>$null
if (-not $smi) { Write-Error "nvidia-smi failed" }
$parts = ($smi -split "," | ForEach-Object { $_.Trim() })
$util = [double]$parts[0]
$memUsed = [double]$parts[1]
$memTotal = [double]$parts[2]
$temp = [double]$parts[3]

$cpuPct = [double](Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
$os = Get-CimInstance Win32_OperatingSystem
$hostMemTotalMb = [math]::Round($os.TotalVisibleMemorySize / 1024, 1)
$hostMemUsedMb = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1024, 1)

$reactionUp = 0
try {
    $h = Invoke-WebRequest -Uri "http://127.0.0.1:8100/health" -UseBasicParsing -TimeoutSec 3
    if ($h.StatusCode -eq 200) { $reactionUp = 1 }
} catch { $reactionUp = 0 }

$ollamaUp = 0
try {
    $o = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 3
    if ($o.StatusCode -eq 200) { $ollamaUp = 1 }
} catch { $ollamaUp = 0 }

$lines = @(
    "# TYPE homelab_gpu_utilization_percent gauge",
    "homelab_gpu_utilization_percent{instance=`"$Instance`",job=`"gpu`"} $util",
    "# TYPE homelab_gpu_memory_used_mib gauge",
    "homelab_gpu_memory_used_mib{instance=`"$Instance`",job=`"gpu`"} $memUsed",
    "# TYPE homelab_gpu_memory_total_mib gauge",
    "homelab_gpu_memory_total_mib{instance=`"$Instance`",job=`"gpu`"} $memTotal",
    "# TYPE homelab_gpu_temperature_c gauge",
    "homelab_gpu_temperature_c{instance=`"$Instance`",job=`"gpu`"} $temp",
    "# TYPE homelab_host_cpu_percent gauge",
    "homelab_host_cpu_percent{instance=`"$Instance`",job=`"gpu`"} $cpuPct",
    "# TYPE homelab_host_memory_used_mib gauge",
    "homelab_host_memory_used_mib{instance=`"$Instance`",job=`"gpu`"} $hostMemUsedMb",
    "# TYPE homelab_host_memory_total_mib gauge",
    "homelab_host_memory_total_mib{instance=`"$Instance`",job=`"gpu`"} $hostMemTotalMb",
    "# TYPE homelab_reaction_serve_up gauge",
    "homelab_reaction_serve_up{instance=`"$Instance`",job=`"gpu`"} $reactionUp",
    "# TYPE homelab_ollama_up gauge",
    "homelab_ollama_up{instance=`"$Instance`",job=`"gpu`"} $ollamaUp"
)
$body = ($lines -join "`n") + "`n"

$uri = "$PushUrl/metrics/job/gpu/instance/$Instance"
Invoke-WebRequest -Uri $uri -Method Put -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) `
    -ContentType "text/plain; version=0.0.4; charset=utf-8" -UseBasicParsing | Out-Null
Write-Output "pushed zak_pc cpu=$cpuPct% ram=${hostMemUsedMb}/${hostMemTotalMb}MiB gpu=$util% reaction=$reactionUp ollama=$ollamaUp"
