param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')][string]$RunId,
    [ValidateSet('smoke','baseline','all')][string]$Stage='baseline',
    [ValidateSet('lublin_256','Cirne','Jann')][string]$Workload='lublin_256',
    [int[]]$Seed=@(0),
    [ValidateSet('cpu','cuda','auto')][string]$Device='cuda',
    [ValidateRange(0,300)][int]$CheckpointEvery=5,
    [ValidateSet('legacy','paired')][string]$EvaluationSeedMode='legacy',
    [string]$ResumeFrom=''
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent (Split-Path -Parent $PSCommandPath)
if (Test-Path -LiteralPath (Join-Path $projectRoot '.research.lock')) {
    throw 'A pipeline owns .research.lock; inspect its actual process tree before any retry.'
}
if (Test-Path -LiteralPath (Join-Path $projectRoot "results\$RunId")) {
    throw 'Run ID already exists. Preserve the old run and choose a new ID.'
}
$python=Join-Path $projectRoot '.venv\Scripts\python.exe'
$script=Join-Path $projectRoot 'scripts\run_research.py'
New-Item -ItemType Directory -Path (Join-Path $projectRoot 'results') -Force | Out-Null
# WMI creates the launcher outside the calling tool job. The venv redirector may
# still create its own job for its interpreter; state.json.pid is authoritative.
# Microsoft: https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
$commandLine='"'+$python+'" -X utf8 "'+$script+'" --stage '+$Stage+' --workloads '+$Workload+' --seeds '+($Seed -join ' ')+' --device '+$Device+' --num-threads 1 --timeout-hours 12 --checkpoint-every '+$CheckpointEvery+' --evaluation-seed-mode '+$EvaluationSeedMode+' --run-id '+$RunId
if ($ResumeFrom) {
    $previous=(Resolve-Path -LiteralPath $ResumeFrom).Path
    $commandLine+=' --resume-from "'+$previous+'"'
}
$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$created=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$commandLine; CurrentDirectory=$projectRoot; ProcessStartupInformation=$startup}
if ($created.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($created.ReturnValue)" }
$record=[ordered]@{created_at_utc=[DateTime]::UtcNow.ToString('o'); run_id=$RunId; launcher_pid=$created.ProcessId; method='Win32_Process.Create'; command=$commandLine; working_directory=$projectRoot; note='Launch accepted; confirm state.json and logs before calling training active.'}
$launchPath=Join-Path $projectRoot "results\launch-$RunId.json"
$record | ConvertTo-Json | Set-Content -LiteralPath $launchPath -Encoding UTF8
$record | ConvertTo-Json -Compress
