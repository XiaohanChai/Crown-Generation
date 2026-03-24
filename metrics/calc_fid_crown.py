import argparse
import json
import os
from pathlib import Path

from cleanfid import fid


def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID for crown views.")
    parser.add_argument("--synthesis_path", required=True, help="Root folder with generated view_* dirs.")
    parser.add_argument("--dataset_path", required=True, help="Root folder with reference view_* dirs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for FID.")
    parser.add_argument("--view_prefix", default="view_", help="Prefix for view folders.")
    parser.add_argument("--out_json", default="", help="Optional JSON output file.")
    return parser.parse_args()


def list_views(root_dir, prefix):
    root_dir = Path(root_dir)
    views = [p.name for p in root_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]

    def sort_key(name):
        suffix = name[len(prefix):]
        try:
            return (0, int(suffix))
        except ValueError:
            return (1, suffix)

    return sorted(views, key=sort_key)


def main():
    args = parse_args()

    views1 = list_views(args.synthesis_path, args.view_prefix)
    views2 = list_views(args.dataset_path, args.view_prefix)

    if views1 != views2:
        missing_in_synth = sorted(set(views2) - set(views1))
        missing_in_ref = sorted(set(views1) - set(views2))
        raise RuntimeError(
            "Mismatched view folders. "
            f"Missing in synthesis: {missing_in_synth}; Missing in reference: {missing_in_ref}"
        )

    fid_sum = 0.0
    fid_dict = {}

    for idx, view in enumerate(views1):
        view1_path = os.path.join(args.synthesis_path, view)
        view2_path = os.path.join(args.dataset_path, view)
        fid_value = fid.compute_fid(view1_path, view2_path, batch_size=args.batch_size)
        fid_sum += fid_value
        fid_dict[view] = fid_value
        print(f"Finish view {idx}")
        print(f"FID for {view}: {fid_value}")

    fid_avg = fid_sum / max(len(views1), 1)
    results = {"FID": fid_avg, "per_view": fid_dict}
    print(json.dumps(results, indent=2, sort_keys=True))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
