"""Small helpers shared by the command-line tools."""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

NUM_BODY_JOINTS = 5  # shoulder_pan .. wrist_roll, in degrees; the gripper (index 5) is on a 0-100 scale


def default_root(repo_id: str) -> Path:
    """Local folder LeRobot uses for a dataset recorded on this machine."""
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id


def read_frames(root: Path, columns: list[str]) -> pd.DataFrame:
    """Read the per-frame table of a LeRobot v3 dataset (no video decoding)."""
    files = sorted(glob.glob(str(Path(root) / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files under {Path(root) / 'data'}")
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


def episode_arrays(df: pd.DataFrame, column: str):
    """Yield (episode_index, array of shape (frames, dims)) in frame order."""
    for ep, g in df.groupby("episode_index"):
        yield int(ep), np.stack(g.sort_values("frame_index")[column].values)


def to_uint8_hwc(img) -> np.ndarray:
    """LeRobot returns float CHW images in [0, 1]; video writers and OpenCV want uint8 HWC."""
    return (img.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
