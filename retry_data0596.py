from pathlib import Path
import numpy as np
import trimesh

from convert_mask_notwatertight_to_watertight import load_mesh, recursive_fix_keep_coords

src = Path('/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_notwatertight/data0596/model.obj')
out = Path('/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_watertight/data0596_watertight.obj')


def check(label: str, path: Path):
    m0 = trimesh.load(path, force='mesh', process=False)
    m1 = trimesh.load(path, force='mesh', process=True)
    print(f'{label}: process=False watertight={bool(m0.is_watertight)}, process=True watertight={bool(m1.is_watertight)}, verts={len(m0.vertices)}, faces={len(m0.faces)}')


def robust_make_watertight_keep_coords(mesh: trimesh.Trimesh):
    orig_center = np.asarray(mesh.bounds).mean(axis=0)
    repaired = recursive_fix_keep_coords(mesh.copy(), max_iter=50)
    if repaired.is_watertight:
        return repaired, 'recursive_fix'

    ext = float(np.max(np.asarray(mesh.bounding_box.extents)))
    pitch = max(ext / 256.0, 1e-5)
    vox = mesh.voxelized(pitch)
    vox = vox.fill()
    mc = vox.marching_cubes
    mc.remove_unreferenced_vertices()
    mc.fix_normals()
    mc_center = np.asarray(mc.bounds).mean(axis=0)
    mc.apply_translation(orig_center - mc_center)
    if mc.is_watertight:
        return mc, f'voxel_marching_cubes(pitch={pitch:.8f})'

    hull = mesh.convex_hull
    hull.remove_unreferenced_vertices()
    hull.fix_normals()
    return hull, 'convex_hull'

print('--- before ---')
check('src', src)
check('out', out)

mesh = load_mesh(src)
if isinstance(mesh, str):
    raise RuntimeError(mesh)

orig_center = np.asarray(mesh.bounds).mean(axis=0)

mesh, method = robust_make_watertight_keep_coords(mesh)

mesh.export(out)

print('--- after retry ---')
check('out', out)
new_mesh = trimesh.load(out, force='mesh', process=False)
new_center = np.asarray(new_mesh.bounds).mean(axis=0)
shift = float(np.linalg.norm(new_center - orig_center))
print(f'method={method}, center_shift={shift:.10f}')
