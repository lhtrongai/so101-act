"""Success rate with a Wilson 95% confidence interval from hand-labeled real-arm trials.

Input: a CSV with one row per trial and at least the columns `checkpoint` and `success`
(1 or 0). Optional columns such as `red_pose` can be used to break the result down.
Notes containing commas must be wrapped in double quotes.

Examples:
  so101-success-rate results/trials_pick_red.csv
  so101-success-rate results/trials_pick_red.csv --by red_pose
"""

import argparse

import pandas as pd
from scipy.stats import binomtest


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(keys):
        n, k = len(g), int(g["success"].sum())
        ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(keys, key)), "trials": n, "successes": k,
                     "rate": f"{k / n:.0%}", "95% CI": f"{ci.low:.0%} - {ci.high:.0%}"})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--by", nargs="*", default=[], help="extra columns to split by, e.g. red_pose")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    missing = df["success"].isna().sum()
    if missing:
        print(f"Skipping {missing} trials without a result yet.")
        df = df.dropna(subset=["success"])
    print(summarize(df, ["checkpoint", *args.by]).to_string(index=False))


if __name__ == "__main__":
    main()
