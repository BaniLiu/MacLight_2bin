import argparse
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = ["Return", "waiting_list", "queue_list", "speed_list"]


def load_final_rows(scene_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in scene_dir.glob("*/*.csv"):
        data = pd.read_csv(csv_path)
        if data.empty:
            continue
        final_row = data.iloc[-1].copy()
        final_row["Source"] = str(csv_path)
        rows.append(final_row)
    if not rows:
        raise FileNotFoundError(f"No result csv files found under {scene_dir}")
    return pd.DataFrame(rows)


def summarize(final_rows: pd.DataFrame) -> pd.DataFrame:
    summary = final_rows.groupby("Algorithm")[METRIC_COLUMNS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index().sort_values("waiting_list_mean")


def main():
    parser = argparse.ArgumentParser(description="Compare model results for one MacLight scene.")
    parser.add_argument("--scene", default="block_normal", help="Scene folder under data/plot_data")
    parser.add_argument("--data-root", default="data/plot_data", help="Root folder containing scene outputs")
    parser.add_argument("--out-dir", default="data/comparison", help="Folder for comparison summaries")
    args = parser.parse_args()

    scene_dir = Path(args.data_root) / args.scene
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_rows = load_final_rows(scene_dir)
    summary = summarize(final_rows)

    final_rows.to_csv(out_dir / f"{args.scene}_final_rows.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / f"{args.scene}_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
