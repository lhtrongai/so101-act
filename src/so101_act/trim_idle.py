"""Trim the idle segments at the start and end of every episode of a LeRobot dataset.

Motion starts at the first frame where any body joint of the action moves more than
--move-threshold-deg from frame 0, and ends at the last frame where any body joint is more
than that from the final frame. The script keeps --head-keep frames of rest before motion
(so the policy sees how to start) and --tail-keep frames after it (so it learns to stop when
done), and writes the result to a new dataset. The source dataset is never modified.

If a camera dropped a frame, LeRobot raises FrameTimestampError for that index; the episode
is ended there rather than pairing joints with the wrong image.

Example:
  so101-trim --src-repo-id tronghle/so101_pick_red_20261110_013629 --dst-repo-id tronghle/so101_pick_red_trimmed
"""

import argparse
from pathlib import Path

import numpy as np

from so101_act.common import NUM_BODY_JOINTS, default_root, read_frames, to_uint8_hwc


def keep_ranges(root: Path, threshold: float, head_keep: int, tail_keep: int) -> dict[int, tuple[int, int]]:
    """{episode: (first frame to keep, frame after the last one to keep)} from the action column."""
    df = read_frames(root, ["episode_index", "frame_index", "action"])
    ranges = {}
    for ep, g in df.groupby("episode_index"):
        body = np.stack(g.sort_values("frame_index")["action"].values)[:, :NUM_BODY_JOINTS]
        moved = np.abs(body - body[0]).max(axis=1) > threshold
        move_start = int(np.argmax(moved)) if moved.any() else 0
        away = np.abs(body - body[-1]).max(axis=1) > threshold
        move_end = len(body) - 1 - int(np.argmax(away[::-1])) if away.any() else len(body) - 1
        ranges[int(ep)] = (max(move_start - head_keep, 0), min(move_end + 1 + tail_keep, len(body)))
    return ranges


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-repo-id", required=True)
    p.add_argument("--src-root", type=Path)
    p.add_argument("--dst-repo-id", required=True)
    p.add_argument("--dst-root", type=Path)
    p.add_argument("--move-threshold-deg", type=float, default=2.0)
    p.add_argument("--head-keep", type=int, default=15, help="rest frames kept before motion (15 = 0.5 s at 30 fps)")
    p.add_argument("--tail-keep", type=int, default=30, help="rest frames kept after motion (30 = 1 s at 30 fps)")
    p.add_argument("--dry-run", action="store_true", help="print what would be cut, write nothing")
    args = p.parse_args()

    src_root = args.src_root or default_root(args.src_repo_id)
    dst_root = args.dst_root or default_root(args.dst_repo_id)
    keep = keep_ranges(src_root, args.move_threshold_deg, args.head_keep, args.tail_keep)
    if args.dry_run:
        for ep, (first, stop) in keep.items():
            print(f"episode {ep:3d}: keep frames {first}..{stop - 1}")
        return
    if dst_root.exists():
        raise SystemExit(f"{dst_root} already exists. Delete it first to rebuild.")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import FrameTimestampError
    from lerobot.utils.constants import DEFAULT_FEATURES

    src = LeRobotDataset(args.src_repo_id, root=src_root)
    camera_keys = set(src.meta.camera_keys)
    copy_keys = [k for k in src.meta.features if k not in DEFAULT_FEATURES]
    dst = LeRobotDataset.create(
        repo_id=args.dst_repo_id,
        fps=src.meta.fps,
        features=src.meta.features,
        root=dst_root,
        robot_type=src.meta.robot_type,
        use_videos=True,
        streaming_encoding=True,
        encoder_threads=2,
    )

    fps = src.meta.fps
    total_kept = 0
    for ep in range(src.meta.total_episodes):
        ep_start = int(src.meta.episodes[ep]["dataset_from_index"])
        ep_len = int(src.meta.episodes[ep]["dataset_to_index"]) - ep_start
        first, stop = keep[ep]
        start, end = ep_start + first, ep_start + stop
        for idx in range(start, end):
            try:
                item = src[idx]
            except FrameTimestampError:
                print(f"episode {ep:3d}: frame {idx - ep_start} missing in video, dropped last {end - idx} frames")
                end = idx
                break
            frame = {k: to_uint8_hwc(item[k]) if k in camera_keys else item[k] for k in copy_keys}
            frame["task"] = item["task"]
            dst.add_frame(frame)
        dst.save_episode()
        total_kept += end - start
        print(
            f"episode {ep:3d}: cut {first / fps:4.1f} s at start, "
            f"{(ep_start + ep_len - end) / fps:3.1f} s at end, kept {end - start} frames"
        )

    dst.finalize()
    print(f"Done. {dst.meta.total_episodes} episodes, {total_kept} frames in {dst_root}")


if __name__ == "__main__":
    main()
