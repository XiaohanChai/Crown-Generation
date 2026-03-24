import argparse
import os
from pathlib import Path

import trimesh

from utils.render_utils import generate_image_for_fid


def parse_args():
    parser = argparse.ArgumentParser(description="Render crown meshes to FID image folders.")
    parser.add_argument("--filelist", required=True, help="Text file with relative mesh paths (no extension).")
    parser.add_argument("--mesh_root", required=True, help="Root folder containing meshes.")
    parser.add_argument("--image_root", required=True, help="Output root folder for FID images.")
    return parser.parse_args()


def read_filelist(filelist_path):
    with open(filelist_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    args = parse_args()

    mesh_root = Path(args.mesh_root)
    image_root = Path(args.image_root)
    image_root.mkdir(parents=True, exist_ok=True)

    rel_paths = read_filelist(args.filelist)

    for idx, rel in enumerate(rel_paths):
        mesh_path = mesh_root / f"{rel}.obj"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing mesh: {mesh_path}")

        mesh = trimesh.load(str(mesh_path), force="mesh")
        generate_image_for_fid(mesh, str(image_root), idx)
        print(f"Finished rendering mesh {idx}")


if __name__ == "__main__":
    main()
