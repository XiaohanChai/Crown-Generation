import argparse
import re
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
    """Repair mesh without changing global coordinates."""
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


def process(src_dir: Path, out_dir: Path, report_path: Path, max_iter: int = 20):
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = 0
    ok_export = 0
    watertight_true = 0
    failed = 0

    pattern = re.compile(r'^(data\d+)_mask\.obj$')
    files = sorted([p for p in src_dir.glob('*_mask.obj') if pattern.match(p.name)])

    for src_path in files:
        total += 1
        m = pattern.match(src_path.name)
        data_id = m.group(1)
        out_path = out_dir / f'{data_id}_mask_watertight.obj'

        mesh = load_mesh(src_path)
        if isinstance(mesh, str):
            rows.append(f"{src_path.name}: {mesh}, watertight: False, saved: None")
            failed += 1
            continue

        try:
            if not mesh.is_watertight:
                mesh = recursive_fix_keep_coords(mesh, max_iter=max_iter)
            mesh.export(out_path)

            is_wt = bool(mesh.is_watertight)
            rows.append(f"{src_path.name}: ok, watertight: {is_wt}, saved: {out_path}")
            ok_export += 1
            if is_wt:
                watertight_true += 1
        except Exception as exc:
            rows.append(f"{src_path.name}: fix/export error: {exc}, watertight: False, saved: None")
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
        '--src-dir',
        type=Path,
        default=Path('/root/octfusion/data/mask_crown/crown_585/mesh_obj/basic_files/mask'),
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('/root/octfusion/data/mask_crown/crown_585/mask_watertight'),
    )
    parser.add_argument(
        '--report',
        type=Path,
        default=Path('/root/octfusion/mesh_watertight_fix_report_basic_files_mask.txt'),
    )
    parser.add_argument('--max-iter', type=int, default=20)
    args = parser.parse_args()

    summary = process(args.src_dir, args.out_dir, args.report, max_iter=args.max_iter)
    print("\n".join(summary))
    print(f"report: {args.report}")


if __name__ == '__main__':
    main()
