"""Run a real-arm evaluation session trial by trial and record the scores.

Reads a trials CSV prepared in advance (columns checkpoint, trial, red_cell, black_cell,
red_pose, success, notes) and goes through the rows of one checkpoint label whose `success`
is still empty. For each trial it:

  1. prints the start configuration and waits for Enter while you set up the scene,
  2. runs one `lerobot-rollout` with the given policy,
  3. asks for the result (1 or 0) and an optional note, and saves the CSV right away.

Stopping half way is safe: rerunning continues from the first unscored trial. A person must
stay next to the arm the whole time; Ctrl+C stops the current rollout.

Example (Windows, inside the LeRobot environment):
  so101-eval-session results/trials_pick_red.csv --label aug_camera_shifted \\
      --policy tronghle/act_so101_pick_red_aug --front 0 --wrist 2
"""

import argparse
import subprocess
from pathlib import Path

import pandas as pd

POSES = {"stand": "standing", "lying": "lying on its side", "tilted": "standing tilted"}


def rollout_command(args) -> list[str]:
    cameras = (
        f"{{ front: {{type: opencv, index_or_path: {args.front}, width: 640, height: 480, fps: 30, fourcc: MJPG}}, "
        f"wrist: {{type: opencv, index_or_path: {args.wrist}, width: 640, height: 480, fps: 30, fourcc: MJPG}}}}"
    )
    cmd = [
        "lerobot-rollout", "--strategy.type=base", f"--policy.path={args.policy}", "--device=cpu",
        "--robot.type=so101_follower", f"--robot.port={args.port}", f"--robot.id={args.robot_id}",
        f"--robot.cameras={cameras}", f"--task={args.task}", f"--duration={args.duration}", "--play_sounds=false",
    ]
    if args.n_action_steps:
        cmd.append(f"--policy.n_action_steps={args.n_action_steps}")
    return cmd


def ask_result() -> tuple[int, str]:
    while True:
        answer = input("Result: 1 = success, 0 = failure, r = redo (setup mistake only, not a policy failure) > ").strip().lower()
        if answer in ("0", "1", "r"):
            break
    if answer == "r":
        return -1, ""
    note = input("Note (Enter to skip) > ").strip()
    return int(answer), note


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path)
    p.add_argument("--label", required=True, help="value of the checkpoint column to run, e.g. aug_camera_shifted")
    p.add_argument("--policy", required=True)
    p.add_argument("--front", type=int, required=True, help="camera index of the front camera")
    p.add_argument("--wrist", type=int, required=True, help="camera index of the wrist camera")
    p.add_argument("--port", default="COM5")
    p.add_argument("--robot-id", default="follower_01")
    p.add_argument("--task", default="Pick the red object and place it in the box")
    p.add_argument("--duration", type=int, default=35)
    p.add_argument("--n-action-steps", type=int, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.csv, dtype={"notes": "string", "success": "Int64"})
    todo = df[(df["checkpoint"] == args.label) & df["success"].isna()]
    if todo.empty:
        raise SystemExit(f"No unscored trials labelled {args.label!r} in {args.csv}. Add the rows first.")
    cmd = rollout_command(args)
    print(f"{len(todo)} trials to run for {args.label}. Score within the first 30 s.\n")

    for i in todo.index:
        row = df.loc[i]
        while True:
            pose = POSES.get(str(row["red_pose"]), str(row["red_pose"]))
            print(f"--- Trial {int(row['trial'])}: red in cell {int(row['red_cell'])} ({pose}), "
                  f"black in cell {int(row['black_cell'])}. Arm at rest, nothing else in the front view.")
            input("Press Enter to start the rollout > ")
            subprocess.run(cmd, check=False)
            success, note = ask_result()
            if success >= 0:
                break
        df.loc[i, "success"] = success
        df.loc[i, "notes"] = note or pd.NA
        df.to_csv(args.csv, index=False)
        done = df[(df["checkpoint"] == args.label) & df["success"].notna()]
        print(f"Saved. {int(done['success'].sum())}/{len(done)} successes so far.\n")

    print("Session complete. Summary: so101-success-rate", args.csv)


if __name__ == "__main__":
    main()
