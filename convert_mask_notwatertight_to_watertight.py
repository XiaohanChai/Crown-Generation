import argparse
from pathlib import Path

import trimesh


def load_mesh(obj_path: Path):
    """Load mesh robustly while preserving coordinates."""
    last_err = None
    for process in (False, True):
        try:
            mesh = trimesh.load(obj_path, force='mesh', process=process)
            if isinstance(mesh, trimesh.Trimesh):
                return mesh
            if hasattr(mesh, "dump"):
                dumped = mesh.dump(concatenate=True)
                if isinstance(dumped, trimesh.Trimesh):
                    return dumped
        except Exception as exc:
            last_err = exc
    return f"load failed: {last_err}"


def recursive_fix_keep_coords(mesh: trimesh.Trimesh, max_iter: int = 20):
    """Repair mesh toward watertightness without global recentering."""
    for _ in range(max_iter):
        mesh.merge_vertices()
        mesh.fill_holes()

        # Keep global coordinates unchanged: do NOT call mesh.rezero().
        if hasattr(mesh, "update_faces") and hasattr(mesh, "nondegenerate_faces"):
            mesh.update_faces(mesh.nondegenerate_faces())
        else:
            mesh.remove_degenerate_faces()

        if hasattr(mesh, "unique_faces"):
            mesh.update_faces(mesh.unique_faces())
        else:
            mesh.remove_duplicate_faces()

        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()

        if mesh.is_watertight:
            break

    return mesh


def process(src_root: Path, out_root: Path, report_path: Path, max_iter: int = 20):
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    total = 0
    ok_export = 0
    watertight_true = 0
    failed = 0

    subfolders = sorted(
        p.name for p in src_root.iterdir() if p.is_dir() and p.name.startswith("data")
    )

    for sub in subfolders:
        total += 1
        obj_path = src_root / sub / "model.obj"
        out_path = out_root / f"{sub}_watertight.obj"

        if not obj_path.exists():
            rows.append(f"{sub}: Not found, watertight: False, saved: None")
            failed += 1
            continue

        mesh = load_mesh(obj_path)
        if isinstance(mesh, str):
            rows.append(f"{sub}: {mesh}, watertight: False, saved: None")
            failed += 1
            continue

        try:
            if not mesh.is_watertight:
                mesh = recursive_fix_keep_coords(mesh, max_iter=max_iter)
            mesh.export(out_path)

            is_wt = bool(mesh.is_watertight)
            rows.append(f"{sub}: ok, watertight: {is_wt}, saved: {out_path}")
            ok_export += 1
            if is_wt:
                watertight_true += 1
        except Exception as exc:
            rows.append(f"{sub}: fix/export error: {exc}, watertight: False, saved: None")
            failed += 1

    summary = [
        f"total: {total}",
        f"ok_export: {ok_export}",
        f"watertight_true: {watertight_true}",
        f"failed: {failed}",
    ]

    report_path.write_text("\n".join(summary + [""] + rows), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_notwatertight"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_watertight"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/root/octfusion/mesh_watertight_fix_report_mask_notwatertight.txt"),
    )
    parser.add_argument("--max-iter", type=int, default=20)
    args = parser.parse_args()

    summary = process(args.src_root, args.out_root, args.report, max_iter=args.max_iter)
    print("\n".join(summary))
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
