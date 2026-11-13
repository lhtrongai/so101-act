"""Check that the cameras still see the scene the way they did during recording.

Three subcommands:

  align     live view of one camera blended 50/50 with a recorded frame, shown in Rerun;
            move the camera (or table) until the white square and the box stop doubling
  compare   one snapshot per camera against a recorded frame: pixel shift, mean colour,
            and a side-by-side image (recorded | live | blend)
  sessions  first front frame of every episode against episode 0, to find recording
            sessions where the camera moved (no hardware needed)

Pixel shifts come from phase correlation. Trust them only when the reported match is
well above about 0.3; different lighting or a different arm pose lowers it.

Examples:
  so101-camera align --repo-id tronghle/so101_pick_red_20261110_013629 --episode 2 --index 0
  so101-camera align --ref-image camera_check/front_color.png --ref-panel 1 --index 0
  so101-camera compare --repo-id tronghle/so101_pick_red_20261110_013629 --episode 2 --front 0 --wrist 2
  so101-camera sessions --repo-id tronghle/so101_pick_red_20261110_013629
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from so101_act.common import default_root, to_uint8_hwc

FRONT, WRIST = "observation.images.front", "observation.images.wrist"


def open_camera(index: int):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def grab(index: int, warmup: int = 30) -> np.ndarray:
    """One RGB frame after letting auto exposure settle."""
    cap = open_camera(index)
    frame = None
    for _ in range(warmup):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        raise RuntimeError(f"Could not read camera {index}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def shift(a_rgb: np.ndarray, b_rgb: np.ndarray) -> tuple[float, float, float]:
    """(dx, dy, match) of b relative to a; positive dx means b is shifted right."""
    ga = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    (dx, dy), match = cv2.phaseCorrelate(ga, gb)
    return dx, dy, match


def load_dataset(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(args.repo_id, root=args.root or default_root(args.repo_id))


def first_frame(ds, episode: int, key: str) -> np.ndarray:
    idx = int(ds.meta.episodes[episode]["dataset_from_index"])
    return to_uint8_hwc(ds[idx][key])


def load_reference(args) -> np.ndarray:
    """Reference frame: a saved image (optionally one 640 px panel of a side-by-side image) or a recorded frame."""
    if args.ref_image:
        img = cv2.imread(str(args.ref_image))
        if img is None:
            raise SystemExit(f"Could not read {args.ref_image}")
        if args.ref_panel is not None:
            img = img[:, 640 * args.ref_panel : 640 * (args.ref_panel + 1)]
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return first_frame(load_dataset(args), args.episode, args.key)


def cmd_align(args) -> None:
    import rerun as rr

    ref = load_reference(args)
    cap = open_camera(args.index)
    rr.init("camera_alignment", spawn=True)
    print("Align until nothing looks doubled in Rerun. Ctrl+C to stop.")
    try:
        while True:
            ok, live = cap.read()
            if ok:
                rr.log("blend", rr.Image(cv2.addWeighted(ref, 0.5, cv2.cvtColor(live, cv2.COLOR_BGR2RGB), 0.5, 0)))
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


def cmd_compare(args) -> None:
    ds = load_dataset(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, key, index in [("front", FRONT, args.front), ("wrist", WRIST, args.wrist)]:
        rec, live = first_frame(ds, args.episode, key), grab(index)
        dx, dy, match = shift(rec, live)
        print(f"{name}: live shifted {dx:+.1f} px right, {dy:+.1f} px down (match {match:.2f})")
        print(f"{name}: mean RGB recorded {rec.reshape(-1, 3).mean(0).round(0)}, live {live.reshape(-1, 3).mean(0).round(0)}")
        panel = np.hstack([rec, live, cv2.addWeighted(rec, 0.5, live, 0.5, 0)])
        cv2.imwrite(str(out / f"{name}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    print(f"Images (recorded | live | blend) saved in {out}")


def cmd_sessions(args) -> None:
    ds = load_dataset(args)
    frames = {ep: first_frame(ds, ep, FRONT) for ep in range(ds.meta.total_episodes)}
    ref = frames[args.reference]
    print(f"First front frame of each episode vs episode {args.reference} (px right, px down, match):")
    for ep, img in frames.items():
        dx, dy, match = shift(ref, img)
        print(f"ep {ep:3d}: {dx:+5.1f} {dy:+5.1f} ({match:.2f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["align", "compare", "sessions"]:
        s = sub.add_parser(name)
        s.add_argument("--repo-id", required=(name != "align"))
        s.add_argument("--root", type=Path)
        if name in ("align", "compare"):
            s.add_argument("--episode", type=int, default=0, help="episode whose first frame is the reference")
    sub.choices["align"].add_argument("--index", type=int, required=True, help="live camera index")
    sub.choices["align"].add_argument("--key", default=FRONT, help="dataset image key to align against")
    sub.choices["align"].add_argument("--ref-image", type=Path, help="align to a saved image instead of a recorded frame")
    sub.choices["align"].add_argument("--ref-panel", type=int, help="panel index (0, 1, 2) when the image is side by side")
    sub.choices["compare"].add_argument("--front", type=int, required=True, help="live index of the front camera")
    sub.choices["compare"].add_argument("--wrist", type=int, required=True, help="live index of the wrist camera")
    sub.choices["compare"].add_argument("--out", default="camera_check")
    sub.choices["sessions"].add_argument("--reference", type=int, default=0)
    args = p.parse_args()
    {"align": cmd_align, "compare": cmd_compare, "sessions": cmd_sessions}[args.cmd](args)


if __name__ == "__main__":
    main()
