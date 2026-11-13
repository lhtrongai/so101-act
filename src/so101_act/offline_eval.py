"""Run a trained policy on recorded frames and compare its actions with the demonstrations.

No robot needed. Every --chunk-stride frames the policy sees the recorded images and joint
state, predicts a full action chunk, and the chunk is compared with what the operator did.
This separates "the policy learned the task" from "the live scene differs from training".

--perturb applies image changes before prediction, to measure how sensitive the policy is
to lighting or to a moved camera without touching the hardware:
  front_bright   front image 7% brighter
  wrist_dark     wrist image 10% darker
  front_shift    front image shifted left by --shift-px

Example:
  so101-offline-eval --policy tronghle/act_so101_pick_red --repo-id tronghle/so101_pick_red_trimmed \\
      --episodes 2 17 20 28 39 50 58 65 --perturb front_bright wrist_dark front_shift
"""

import argparse
from pathlib import Path

import numpy as np

from so101_act.common import default_root, read_frames

FRONT, WRIST = "observation.images.front", "observation.images.wrist"
JOINTS = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]


def shift_left(img, px: int):
    out = img.clone()
    out[..., :-px] = img[..., px:]
    out[..., -px:] = img[..., -1:]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="Hub repo or local folder of a trained ACT policy")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", type=Path)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    p.add_argument("--perturb", nargs="*", default=[], choices=["front_bright", "wrist_dark", "front_shift"])
    p.add_argument("--shift-px", type=int, default=20)
    args = p.parse_args()

    import torch

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    root = args.root or default_root(args.repo_id)
    policy = ACTPolicy.from_pretrained(args.policy)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=args.policy, preprocessor_overrides={"device_processor": {"device": "cpu"}}
    )
    ds = LeRobotDataset(args.repo_id, root=root)
    actions = read_frames(root, ["index", "action"]).set_index("index")["action"]
    chunk = policy.config.chunk_size

    changes = {"as recorded": lambda o: o}
    if "front_bright" in args.perturb:
        changes["front 7% brighter"] = lambda o: {**o, FRONT: (o[FRONT] * 1.07).clamp(0, 1)}
    if "wrist_dark" in args.perturb:
        changes["wrist 10% darker"] = lambda o: {**o, WRIST: o[WRIST] * 0.90}
    if "front_shift" in args.perturb:
        changes[f"front {args.shift_px} px left"] = lambda o: {**o, FRONT: shift_left(o[FRONT], args.shift_px)}

    samples = []
    for ep in args.episodes:
        start = int(ds.meta.episodes[ep]["dataset_from_index"])
        end = int(ds.meta.episodes[ep]["dataset_to_index"])
        for idx in range(start, end - chunk, chunk):
            item = ds[idx]
            obs = {k: item[k].unsqueeze(0) for k in ["observation.state", FRONT, WRIST]}
            samples.append((ep, obs, item["task"], np.stack(actions.loc[idx : idx + chunk - 1].values)))

    print("mean |predicted - recorded| per joint (degrees; gripper on 0-100), and signed pan bias")
    print(f"{'condition':<22} " + " ".join(f"{j:>6}" for j in JOINTS) + "  pan bias   pan bias per episode")
    for name, change in changes.items():
        per_ep, errs = {}, []
        for ep, obs, task, gt in samples:
            o = {k: v.clone() for k, v in change(dict(obs)).items()}
            o["task"] = [task]
            with torch.no_grad():
                pred = post(policy.predict_action_chunk(pre(o)))[0].numpy()
            errs.append(np.abs(pred - gt))
            per_ep.setdefault(ep, []).append(pred[:, 0] - gt[:, 0])
        err = np.concatenate(errs).mean(axis=0)
        bias = {ep: np.concatenate(v).mean() for ep, v in per_ep.items()}
        print(
            f"{name:<22} " + " ".join(f"{e:6.2f}" for e in err)
            + f"  {np.mean(list(bias.values())):+8.2f}   " + " ".join(f"{b:+5.1f}" for b in bias.values())
        )


if __name__ == "__main__":
    main()
