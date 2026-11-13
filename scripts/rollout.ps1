# Run a trained policy on the SO-101 follower for one evaluation trial (Windows PowerShell).
# Check camera indices first: lerobot-find-cameras opencv, then look at the saved images.
# Usage:
#   .\scripts\rollout.ps1 -Policy tronghle/act_so101_pick_red -Front 0 -Wrist 2
param(
    [Parameter(Mandatory = $true)][string]$Policy,
    [int]$Front = 0,
    [int]$Wrist = 2,
    [string]$Port = "COM5",
    [string]$RobotId = "follower_01",
    [string]$Task = "Pick the red object and place it in the box",
    [int]$Duration = 35
)

$cameras = "{ front: {type: opencv, index_or_path: $Front, width: 640, height: 480, fps: 30, fourcc: MJPG}, " +
           "wrist: {type: opencv, index_or_path: $Wrist, width: 640, height: 480, fps: 30, fourcc: MJPG}}"

lerobot-rollout --strategy.type=base --policy.path=$Policy --device=cpu `
    --robot.type=so101_follower --robot.port=$Port --robot.id=$RobotId `
    --robot.cameras="$cameras" --task="$Task" --duration=$Duration --play_sounds=false
