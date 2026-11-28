param(
    [Parameter(Mandatory = $true)]
    [string]$DropletIP,

    [ValidateSet("preflight", "A", "B", "C", "D")]
    [string]$Session = "preflight",

    [string]$Container = "rocm"
)

$ErrorActionPreference = "Stop"
$RemoteDir = "/root/mi300x_scripts"

Write-Host "Uploading scripts/mi300x to root@${DropletIP}:${RemoteDir}..." -ForegroundColor Cyan
ssh "root@${DropletIP}" "mkdir -p ${RemoteDir} && docker exec ${Container} mkdir -p /workspace/mi300x_scripts"
scp -r scripts/mi300x/* "root@${DropletIP}:${RemoteDir}/"

# Strip Windows CRLF and copy into ROCm container
ssh "root@${DropletIP}" "for f in ${RemoteDir}/*.sh; do tr -d '\r' < \`"\$f\`" > \`"\$f.lf\`" && mv \`"\$f.lf\`" \`"\$f\`" && chmod +x \`"\$f\`"; done; docker cp ${RemoteDir}/. ${Container}:/workspace/mi300x_scripts/"

switch ($Session) {
    "preflight" {
        ssh "root@${DropletIP}" "docker exec ${Container} bash /workspace/mi300x_scripts/preflight.sh preflight"
    }
    "A" {
        ssh "root@${DropletIP}" "docker exec ${Container} bash /workspace/mi300x_scripts/session_a_twin.sh"
    }
    "B" {
        ssh "root@${DropletIP}" "docker exec ${Container} bash /workspace/mi300x_scripts/session_b_phase4.sh"
    }
    "C" {
        ssh "root@${DropletIP}" "docker exec ${Container} bash /workspace/mi300x_scripts/session_c_system2.sh 3B"
    }
    "D" {
        ssh "root@${DropletIP}" "docker exec ${Container} python3 /workspace/mi300x_scripts/session_d_fuzzing.py --episodes 100000 --workers 20"
    }
}
