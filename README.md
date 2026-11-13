# ACT on a real SO-101 arm: data checks, training and evaluation

Imitation learning on a self-assembled SO-101 arm (Feetech STS3215 servos, leader-follower
teleoperation) with [LeRobot](https://github.com/huggingface/lerobot). This repo holds the
tools around LeRobot that turned 72 teleoperated demonstrations into a policy measured on the
real arm, plus the diagnosis of what made it fail.

**Task.** Pick a red silicone hexagon and drop it into a box, with a black hexagon of the same
shape always present as a distractor. Both objects are shuffled across a 3x3 grid of start
cells inside a taped square; the red one stands, lies or leans.

**Result so far (ACT, 100k steps):** 17/20 successes on the real arm, **85% (Wilson 95% CI
64-95%)**, with the front camera about 13-25 px off its recording position. All failures share
one pattern: the grasp lands slightly to the right of the object at the same cells.

## Setup

| Item | Value |
|---|---|
| Robot | SO-101 follower (12 V servos) + leader arm |
| Cameras | front: Logitech C270 on a tripod; wrist: UVC module on the gripper; both 640x480 at 30 fps |
| Dataset | 72 episodes, every ordered (red cell, black cell) pair once, 23,783 frames after trimming |
| Policy | ACT, LeRobot defaults (chunk 100, ResNet18, both cameras, batch 8), 100k steps |
| Training | one rented RTX 3090: 0.124 s/step, 3.5 h, about 2 USD |
| LeRobot | commit `38b975c` (pinned for every machine) |

## Install

On a machine that already has LeRobot (editable install) in its environment:

```
git clone https://github.com/lhtrongai/so101-act.git
cd so101-act
uv pip install -e .
```

On a fresh GPU pod: `bash scripts/setup_pod.sh` (installs LeRobot at the pinned commit too).

## Pipeline

1. **Check the recorded data** (seconds, no video decoding):
   `so101-check-dataset --repo-id <user>/<dataset>`
   Prints per-episode idle time at start and end, joint jumps, leader-follower lag, gripper
   range, end offset and mid-task pauses, compared with the dataset median.

2. **Trim idle segments** into a new dataset (the source is never modified):
   `so101-trim --src-repo-id <user>/<dataset> --dst-repo-id <user>/<dataset>_trimmed`
   Keeps 0.5 s of rest before motion and 1 s after it.

3. **Train** on a GPU pod: benchmark 200 steps, then the full run, then upload all checkpoints
   before stopping the pod:
   ```
   bash scripts/train_act.sh <user>/<dataset>_trimmed <user>/act_<name> 200
   bash scripts/train_act.sh <user>/<dataset>_trimmed <user>/act_<name>
   bash scripts/upload_checkpoints.sh <user>/act_<name>
   ```

4. **Evaluate on the real arm** with a fixed list of 20 start configurations, one fresh run
   per trial, scored by hand in a CSV (`results/trials_pick_red.csv`):
   ```
   .\scripts\rollout.ps1 -Policy <user>/act_<name> -Front 0 -Wrist 2
   so101-success-rate results/trials_pick_red.csv --by red_pose
   ```

5. **Diagnose** without the robot:
   - `so101-offline-eval`: run the policy on recorded frames and compare with the demonstrations,
     optionally with lighting or camera-shift perturbations.
   - `so101-camera sessions|compare|align`: find where the camera moved between recording
     sessions, compare live views with recorded ones, and realign a camera live in Rerun.

## Evaluation protocol

- 20 trials per condition, fixed in advance: red cell cycles 1-9, black cell from a fixed table,
  red pose split evenly over standing, lying and tilted (as in the demonstrations).
- Each trial starts from the rest pose with a fresh `lerobot-rollout` run.
- Success within the first 30 s: red object fully inside the box, gripper released, arm back at
  rest, black object not pushed in. Moving again after that is noted but not a failure.
- Trials outside the training region (red object outside the square, stacked on the black one)
  are kept out of the success rate; the policy does not grasp in either case.

## What the failures turned out to be

| Check | Result |
|---|---|
| Arm hardware | Replaying a recorded episode still grasps correctly |
| Policy on recorded frames | Base-joint error about 1.2 degrees, no side bias; failing cells no worse than good ones |
| Wrist camera mount | Gripper fingers at the same pixels as during recording |
| Lighting | Front 7% brighter or wrist 10% darker (simulated on recorded frames): no change |
| Camera between recording sessions | Moved once by about 5 px (episodes 45-71); robot, square and box moved together, so only the camera moved |
| Front camera position | A 20 px shift (simulated) doubles the base-joint error, 1.17 to 2.53 degrees |

The policy learned the task but is sensitive to where the front camera sits: a few tens of
pixels change the result. Next run: the same data with LeRobot image augmentation on (random
shifts up to 5%, rotation up to 5 degrees, colour jitter), evaluated at two camera positions.

## Lessons kept for the next task

- Long idle segments at the start of demos (here about 7 s, waiting for a spoken cue) teach the
  policy to stay still from the start pose; trim them, but keep about 1 s of rest at the end so
  it learns to stop.
- Low training loss is not success on the arm: keep several checkpoints and pick by real trials.
- Measure the training bottleneck before choosing a GPU (`data_s` against `updt_s` in the log).
- Look at every camera image before a session; indices change when cables are replugged.
- Tape the tripod position and do not move the table between recording and evaluation.

## License

Apache-2.0, the same license as LeRobot. See `LICENSE`.
