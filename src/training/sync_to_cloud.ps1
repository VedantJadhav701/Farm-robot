<#
.SYNOPSIS
  Sync project to MI300X cloud GPU via SSH, or pull trained weights back.

.USAGE
  # Upload project to MI300X:
  .\src\training\sync_to_cloud.ps1 -SshHost "user@your-mi300x-ip" -Action push

  # Pull trained models back to local:
  .\src\training\sync_to_cloud.ps1 -SshHost "user@your-mi300x-ip" -Action pull
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$SshHost,

    [string]$RemoteDir = "~/physical-ai-robotics-venture",

    [ValidateSet("push", "pull", "status")]
    [string]$Action = "push",

    [int]$SshPort = 22
)

$LocalRoot = (Resolve-Path "$PSScriptRoot\..\..").Path

if ($Action -eq "push") {
    Write-Host "[→] Creating remote directory $RemoteDir on $SshHost..."
    ssh -p $SshPort $SshHost "mkdir -p $RemoteDir/src $RemoteDir/data $RemoteDir/models"

    Write-Host "[→] Uploading src/ and data/pomegranate_dataset/data.yaml..."
    scp -P $SshPort -r "$LocalRoot\src" "${SshHost}:${RemoteDir}/"
    ssh -p $SshPort $SshHost "mkdir -p $RemoteDir/data/pomegranate_dataset"
    scp -P $SshPort "$LocalRoot\data\pomegranate_dataset\data.yaml" "${SshHost}:${RemoteDir}/data/pomegranate_dataset/"

    Write-Host "`n[✓] Upload complete! Next, SSH in and run:"
    Write-Host "    ssh -p $SshPort $SshHost"
    Write-Host "    cd $RemoteDir"
    Write-Host "    export ROBOFLOW_API_KEY='YOUR_KEY'"
    Write-Host "    tmux new -s train"
    Write-Host "    bash src/training/cloud_run_all.sh"
}
elseif ($Action -eq "pull") {
    Write-Host "[→] Pulling trained models from $SshHost..."
    New-Item -ItemType Directory -Force -Path "$LocalRoot\models" | Out-Null
    scp -P $SshPort -r "${SshHost}:${RemoteDir}/models/*" "$LocalRoot\models\"
    Write-Host "[✓] Models downloaded to $LocalRoot\models\"
    Write-Host "    Test locally with: python src/test_pomegranate_dosing.py"
}
elseif ($Action -eq "status") {
    ssh -p $SshPort $SshHost "rocm-smi 2>/dev/null || nvidia-smi 2>/dev/null; ls -lh ${RemoteDir}/models 2>/dev/null"
}
