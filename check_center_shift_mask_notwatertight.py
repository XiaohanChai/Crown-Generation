from pathlib import Path
import numpy as np
import trimesh

src_root = Path('/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_notwatertight')
out_root = Path('/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_watertight')

subs = sorted([p.name for p in src_root.iterdir() if p.is_dir() and p.name.startswith('data')])
max_shift = 0.0
max_sub = None
n_gt_1e6 = 0
n_gt_1e4 = 0

for sub in subs:
    src_path = src_root / sub / 'model.obj'
    out_path = out_root / f'{sub}_watertight.obj'
    ms = trimesh.load(src_path, force='mesh', process=False)
    mo = trimesh.load(out_path, force='mesh', process=False)

    cs = np.asarray(ms.bounds).mean(axis=0)
    co = np.asarray(mo.bounds).mean(axis=0)
    shift = float(np.linalg.norm(co - cs))

    if shift > max_shift:
        max_shift = shift
        max_sub = sub
    if shift > 1e-6:
        n_gt_1e6 += 1
    if shift > 1e-4:
        n_gt_1e4 += 1

print(f'total={len(subs)}')
print(f'max_center_shift={max_shift:.10f} at {max_sub}')
print(f'shift_gt_1e-6={n_gt_1e6}')
print(f'shift_gt_1e-4={n_gt_1e4}')
