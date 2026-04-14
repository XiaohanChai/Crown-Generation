import argparse
import json
import math
from pathlib import Path

import numpy as np
import trimesh


def natural_key(path: Path):
    parts = []
    token = ""
    for char in path.stem:
        if char.isdigit():
            token += char
        else:
            if token:
                parts.append(int(token))
                token = ""
            parts.append(char)
    if token:
        parts.append(int(token))
    return parts


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [geometry for geometry in mesh.dump() if isinstance(geometry, trimesh.Trimesh)]
        if not geometries:
            raise ValueError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {path}: {type(mesh)}")
    return mesh


def compute_bbox_transform(source_mesh: trimesh.Trimesh, reference_mesh: trimesh.Trimesh):
    source_min, source_max = source_mesh.bounds
    reference_min, reference_max = reference_mesh.bounds

    source_center = (source_min + source_max) * 0.5
    reference_center = (reference_min + reference_max) * 0.5

    source_extent = source_max - source_min
    reference_extent = reference_max - reference_min

    source_scale = float(np.linalg.norm(source_extent))
    reference_scale = float(np.linalg.norm(reference_extent))

    if math.isclose(source_scale, 0.0):
        scale_factor = 1.0
    else:
        scale_factor = reference_scale / source_scale

    translation = reference_center - source_center * scale_factor

    return {
        "scale_factor": float(scale_factor),
        "translation": translation.astype(float).tolist(),
        "source_center": source_center.astype(float).tolist(),
        "reference_center": reference_center.astype(float).tolist(),
        "source_bbox_min": source_min.astype(float).tolist(),
        "source_bbox_max": source_max.astype(float).tolist(),
        "reference_bbox_min": reference_min.astype(float).tolist(),
        "reference_bbox_max": reference_max.astype(float).tolist(),
        "source_bbox_diagonal": float(source_scale),
        "reference_bbox_diagonal": float(reference_scale),
    }


def apply_transform(mesh: trimesh.Trimesh, scale_factor: float, translation):
    transformed = mesh.copy()
    transformed.vertices = transformed.vertices * scale_factor + np.asarray(translation, dtype=float)
    return transformed


def save_mesh(mesh: trimesh.Trimesh, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def main():
    parser = argparse.ArgumentParser(
        description="Align mesh_pair OBJ files to original OBJ size/position, then apply the same transform to generate OBJ files."
    )
    parser.add_argument("--original_dir", default="/root/octfusion/analysis/original", help="Reference OBJ directory")
    parser.add_argument("--mesh_pair_dir", default="/root/octfusion/analysis/mesh_pair", help="Source OBJ directory to align")
    parser.add_argument("--generate_dir", default="/root/octfusion/analysis/generate", help="Directory that receives the same transform")
    parser.add_argument(
        "--out_mesh_pair_dir",
        default="/root/octfusion/analysis/mesh_pair_aligned",
        help="Output directory for aligned mesh_pair OBJ files",
    )
    parser.add_argument(
        "--out_generate_dir",
        default="/root/octfusion/analysis/generate_aligned",
        help="Output directory for aligned generate OBJ files",
    )
    parser.add_argument(
        "--log_path",
        default="/root/octfusion/analysis/tools/align_transform_log.json",
        help="Path to write transform records",
    )
    args = parser.parse_args()

    original_dir = Path(args.original_dir)
    mesh_pair_dir = Path(args.mesh_pair_dir)
    generate_dir = Path(args.generate_dir)
    out_mesh_pair_dir = Path(args.out_mesh_pair_dir)
    out_generate_dir = Path(args.out_generate_dir)
    log_path = Path(args.log_path)

    original_files = {path.stem: path for path in original_dir.glob("*.obj")}
    mesh_pair_files = {path.stem: path for path in mesh_pair_dir.glob("*.obj")}
    generate_files = {path.stem: path for path in generate_dir.glob("*.obj")}

    common_names = sorted(set(original_files) & set(mesh_pair_files) & set(generate_files), key=lambda name: natural_key(Path(name)))
    missing_original = sorted(set(mesh_pair_files) - set(original_files), key=lambda name: natural_key(Path(name)))
    missing_generate = sorted(set(mesh_pair_files) - set(generate_files), key=lambda name: natural_key(Path(name)))

    if missing_original:
        print(f"Warning: {len(missing_original)} mesh_pair files have no original reference: {', '.join(missing_original[:10])}")
    if missing_generate:
        print(f"Warning: {len(missing_generate)} mesh_pair files have no generate counterpart: {', '.join(missing_generate[:10])}")

    records = {}
    processed = 0

    for name in common_names:
        reference_mesh = load_mesh(original_files[name])
        source_mesh = load_mesh(mesh_pair_files[name])
        generate_mesh = load_mesh(generate_files[name])

        transform = compute_bbox_transform(source_mesh, reference_mesh)
        scale_factor = transform["scale_factor"]
        translation = transform["translation"]

        aligned_source = apply_transform(source_mesh, scale_factor, translation)
        aligned_generate = apply_transform(generate_mesh, scale_factor, translation)

        save_mesh(aligned_source, out_mesh_pair_dir / f"{name}.obj")
        save_mesh(aligned_generate, out_generate_dir / f"{name}.obj")

        records[name] = transform
        processed += 1

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Processed {processed} meshes")
    print(f"Aligned mesh_pair saved to: {out_mesh_pair_dir}")
    print(f"Aligned generate saved to: {out_generate_dir}")
    print(f"Transform log saved to: {log_path}")


if __name__ == "__main__":
    main()