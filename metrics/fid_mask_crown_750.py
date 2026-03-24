import argparse
import json
import os
import sys
from pathlib import Path

import trimesh
from cleanfid import fid

# Ensure repo root is on sys.path when running as a script.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.render_utils import generate_image_for_fid


def read_filelist(filelist_path):
    with open(filelist_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def safe_id(name):
    return name.replace("/", "__")


def build_ref_items(filelist_path, mesh_root):
    rel_paths = read_filelist(filelist_path)
    mesh_root = Path(mesh_root)
    items = []
    for rel in rel_paths:
        mesh_path = mesh_root / f"{rel}.obj"
        items.append((safe_id(rel), mesh_path))
    return items


def list_gen_items(gen_dir):
    gen_dir = Path(gen_dir)
    meshes = [p for p in gen_dir.iterdir() if p.suffix.lower() == ".obj"]
    def sort_key(p):
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.stem)
    meshes.sort(key=sort_key)
    return [(p.stem, p) for p in meshes]


def render_items(items, out_dir, render_resolution=299):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for item_id, mesh_path in items:
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing mesh: {mesh_path}")
        mesh = trimesh.load(str(mesh_path), force="mesh")
        generate_image_for_fid(mesh, str(out_dir), item_id)


def compute_fid_for_views(gen_fid_dir, ref_fid_dir, batch_size):
    gen_fid_dir = Path(gen_fid_dir)
    ref_fid_dir = Path(ref_fid_dir)

    gen_views = sorted([p for p in gen_fid_dir.iterdir() if p.is_dir() and p.name.startswith("view_")])
    ref_views = sorted([p for p in ref_fid_dir.iterdir() if p.is_dir() and p.name.startswith("view_")])

    gen_view_names = {p.name for p in gen_views}
    ref_view_names = {p.name for p in ref_views}
    common_views = sorted(gen_view_names & ref_view_names)
    if not common_views:
        raise RuntimeError("No matching view_* folders found for FID computation.")

    fid_sum = 0.0
    fid_dict = {}
    for view in common_views:
        view1_path = str(gen_fid_dir / view)
        view2_path = str(ref_fid_dir / view)
        fid_value = fid.compute_fid(view1_path, view2_path, batch_size=batch_size)
        fid_sum += fid_value
        fid_dict[view] = fid_value

    fid_avg = fid_sum / len(common_views)
    return fid_avg, fid_dict


def main():
    parser = argparse.ArgumentParser(description="Render crown meshes and compute FID.")
    parser.add_argument("--gen_dir", required=True, help="Directory with generated .obj meshes.")
    parser.add_argument("--ref_list", required=True, help="Filelist of reference meshes.")
    parser.add_argument("--ref_mesh_root", required=True, help="Root folder that contains reference meshes.")
    parser.add_argument("--fid_root", required=True, help="Output root dir for FID images.")
    parser.add_argument("--gen_tag", default="gen", help="Subfolder name for generated images.")
    parser.add_argument("--ref_tag", default="ref", help="Subfolder name for reference images.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for FID.")
    parser.add_argument("--max_gen", type=int, default=0, help="Limit number of generated meshes (0 = all).")
    parser.add_argument("--max_ref", type=int, default=0, help="Limit number of reference meshes (0 = all).")
    parser.add_argument("--skip_render_gen", action="store_true", help="Skip rendering generated images.")
    parser.add_argument("--skip_render_ref", action="store_true", help="Skip rendering reference images.")
    parser.add_argument("--out_json", default="", help="Output JSON file for FID.")
    args = parser.parse_args()

    fid_root = Path(args.fid_root)
    gen_fid_dir = fid_root / args.gen_tag
    ref_fid_dir = fid_root / args.ref_tag

    gen_items = list_gen_items(args.gen_dir)
    if args.max_gen > 0:
        gen_items = gen_items[: args.max_gen]

    ref_items = build_ref_items(args.ref_list, args.ref_mesh_root)
    if args.max_ref > 0:
        ref_items = ref_items[: args.max_ref]

    if not args.skip_render_gen:
        render_items(gen_items, gen_fid_dir)

    if not args.skip_render_ref:
        render_items(ref_items, ref_fid_dir)

    fid_avg, fid_dict = compute_fid_for_views(gen_fid_dir, ref_fid_dir, args.batch_size)
    results = {"FID": fid_avg, "per_view": fid_dict}
    print(json.dumps(results, indent=2, sort_keys=True))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
