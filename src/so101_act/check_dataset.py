"""Per-episode data checks for a teleoperated LeRobot dataset.

Reads only the joint table (no video), so it runs in seconds. Prints, per episode, how long
the arm sits still at the start and end, and six signals that point to a bad demonstration:

  duration_s            episode length
  max_jump_deg          largest body-joint change between two consecutive frames (sensor glitch)
  max_lag_deg           largest gap between leader (action) and follower (state) (stall, overload)
  gripper_range         how far the gripper moved (0 means nothing was grasped)
  end_offset_deg        distance of the last frame from the first (did the arm return to rest)
  longest_pause_s       longest still segment after motion started (operator stopped mid-task)

Episodes are compared against the dataset median rather than fixed thresholds: open the
videos of the episodes that stand out.

Example:
  so101-check-dataset --repo-id tronghle/so101_pick_red_20261110_013629
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from so101_act.common import NUM_BODY_JOINTS, default_root, read_frames


def idle_lengths(body: np.ndarray, threshold: float) -> tuple[int, int]:
    """Frames before motion starts and after it ends (motion = any body joint > threshold from the end frame)."""
    moved = np.abs(body - body[0]).max(axis=1) > threshold
    start = int(np.argmax(moved)) if moved.any() else len(body)
    away = np.abs(body - body[-1]).max(axis=1) > threshold
    last = len(body) - 1 - int(np.argmax(away[::-1])) if away.any() else 0
    return start, len(body) - 1 - last


def longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for x in mask:
        run = run + 1 if x else 0
        best = max(best, run)
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", type=Path, help="dataset folder (default: the local LeRobot cache)")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--move-threshold-deg", type=float, default=2.0)
    p.add_argument("--pause-step-deg", type=float, default=0.3, help="per-frame change below this counts as still")
    p.add_argument("--top", type=int, default=3, help="episodes listed per signal")
    args = p.parse_args()

    root = args.root or default_root(args.repo_id)
    df = read_frames(root, ["episode_index", "frame_index", "action", "observation.state"])
    rows = []
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        action = np.stack(g["action"].values)
        state = np.stack(g["observation.state"].values)
        body = action[:, :NUM_BODY_JOINTS]
        head, tail = idle_lengths(body, args.move_threshold_deg)
        still = np.abs(np.diff(body[head:], axis=0)).max(axis=1) < args.pause_step_deg if head < len(body) - 1 else []
        rows.append(
            {
                "episode": int(ep),
                "duration_s": len(body) / args.fps,
                "idle_start_s": head / args.fps,
                "idle_end_s": tail / args.fps,
                "max_jump_deg": np.abs(np.diff(body, axis=0)).max(),
                "max_lag_deg": np.abs(body - state[:, :NUM_BODY_JOINTS]).max(),
                "gripper_range": action[:, -1].max() - action[:, -1].min(),
                "end_offset_deg": np.abs(body[-1] - body[0]).max(),
                "longest_pause_s": longest_run(np.asarray(still)) / args.fps,
            }
        )
    r = pd.DataFrame(rows).round(2)
    pd.set_option("display.width", 160)
    print(f"{len(r)} episodes, {len(df)} frames\n")
    print(r.drop(columns="episode").describe().loc[["min", "50%", "max"]].round(2).to_string())
    for col in ["duration_s", "idle_start_s", "idle_end_s", "max_jump_deg", "max_lag_deg", "end_offset_deg", "longest_pause_s"]:
        top = r.nlargest(args.top, col)[["episode", col]]
        print(f"\nlargest {col}: " + ", ".join(f"ep {int(e)} ({float(v):.2f})" for e, v in top.values))
    low = r.nsmallest(args.top, "gripper_range")[["episode", "gripper_range"]]
    print("\nsmallest gripper_range: " + ", ".join(f"ep {int(e)} ({float(v):.2f})" for e, v in low.values))


if __name__ == "__main__":
    main()
