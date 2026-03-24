import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

# Ensure repo root is on sys.path when running as a script.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from metrics.evaluation_metrics import compute_cov_mmd, compute_1_nna


def scale_to_unit_cube(mesh, padding=0.0):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents) * (1 - padding)
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def sample_points(mesh_path, num_samples):
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh = scale_to_unit_cube(mesh)
    points = mesh.sample(count=num_samples)
    return points.astype(np.float32)


def read_filelist(filelist_path):
    with open(filelist_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def build_ref_paths(filelist_path, mesh_root):
    rel_paths = read_filelist(filelist_path)
    mesh_root = Path(mesh_root)
    mesh_paths = [mesh_root / f"{rel}.obj" for rel in rel_paths]
    return mesh_paths


def _try_parse_int(name):
    try:
        return int(name)
    except ValueError:
        return None


def list_meshes(mesh_dir):
    mesh_dir = Path(mesh_dir)
    meshes = [p for p in mesh_dir.iterdir() if p.suffix.lower() == ".obj"]
    def sort_key(p):
        parsed = _try_parse_int(p.stem)
        return (parsed is None, parsed if parsed is not None else p.stem)
    meshes.sort(key=sort_key)
    return meshes


def load_or_build_pcs(paths, num_samples, cache_path=None, seed=0):
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["pcs"]

    np.random.seed(seed)
    pcs = []
    for mesh_path in paths:
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Missing mesh: {mesh_path}")
        pcs.append(sample_points(str(mesh_path), num_samples))
    pcs = np.stack(pcs, axis=0)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, pcs=pcs)

    return pcs


def main():
    parser = argparse.ArgumentParser(description="Evaluate mask_crown_750 results with point-cloud metrics.")
    parser.add_argument("--gen_dir", required=True, help="Directory with generated .obj meshes.")
    parser.add_argument("--ref_list", required=True, help="Filelist of reference meshes.")
    parser.add_argument("--ref_mesh_root", required=True, help="Root folder that contains reference meshes.")
    parser.add_argument("--num_samples", type=int, default=2048, help="Points sampled per mesh.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for metric computation.")
    parser.add_argument("--max_gen", type=int, default=0, help="Limit number of generated meshes (0 = all).")
    parser.add_argument("--max_ref", type=int, default=0, help="Limit number of reference meshes (0 = all).")
    parser.add_argument("--skip_1nna", action="store_true", help="Skip 1-NN metric (expensive).")
    parser.add_argument("--ref_cache", default="", help="Cache file for reference point clouds (.npz).")
    parser.add_argument("--gen_cache", default="", help="Cache file for generated point clouds (.npz).")
    parser.add_argument("--device", default="", help="Device for metrics (e.g. cuda:0, cpu).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("--out_json", default="", help="Output JSON file for metrics.")
    args = parser.parse_args()

    gen_meshes = list_meshes(args.gen_dir)
    if args.max_gen > 0:
        gen_meshes = gen_meshes[: args.max_gen]

    ref_meshes = build_ref_paths(args.ref_list, args.ref_mesh_root)
    if args.max_ref > 0:
        ref_meshes = ref_meshes[: args.max_ref]

    gen_pcs = load_or_build_pcs(gen_meshes, args.num_samples, args.gen_cache, seed=args.seed)
    ref_pcs = load_or_build_pcs(ref_meshes, args.num_samples, args.ref_cache, seed=args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("Warning: CPU mode is slow for 1-NN and EMD. Consider --skip_1nna or use CUDA.")

    gen_tensor = torch.from_numpy(gen_pcs).to(device=device, dtype=torch.float32)
    ref_tensor = torch.from_numpy(ref_pcs).to(device=device, dtype=torch.float32)

    results = {}
    results.update(compute_cov_mmd(gen_tensor, ref_tensor, batch_size=args.batch_size))

    if not args.skip_1nna:
        results.update(compute_1_nna(gen_tensor, ref_tensor, batch_size=args.batch_size))

    results_out = {k: (v.item() if hasattr(v, "item") else float(v)) for k, v in results.items()}
    print(json.dumps(results_out, indent=2, sort_keys=True))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results_out, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
