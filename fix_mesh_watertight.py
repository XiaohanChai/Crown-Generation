import trimesh
import trimesh
import open3d as o3d
import os

def try_trimesh_load(obj_path):
    try:
        mesh = trimesh.load(obj_path, force='mesh', process=True)
        if hasattr(mesh, 'is_watertight'):
            return mesh
        # 尝试 as_mesh
        if hasattr(mesh, 'as_mesh'):
            mesh = mesh.as_mesh()
            return mesh
    except Exception as e:
        return f'trimesh load error: {e}'
    return 'trimesh load failed'

def try_open3d_load(obj_path):
    try:
        mesh = o3d.io.read_triangle_mesh(obj_path)
        if mesh.is_empty():
            return 'open3d mesh empty'
        # 转trimesh
        vertices = mesh.vertices
        triangles = mesh.triangles
        tm = trimesh.Trimesh(vertices=o3d.utility.Vector3dVector(vertices), faces=o3d.utility.Vector3iVector(triangles))
        return tm
    except Exception as e:
        return f'open3d load error: {e}'


def recursive_fix(mesh, max_iter=10):
    for i in range(max_iter):
        mesh.merge_vertices()
        mesh.fill_holes()
        mesh.rezero()
        if hasattr(mesh, 'update_faces') and hasattr(mesh, 'nondegenerate_faces'):
            mesh.update_faces(mesh.nondegenerate_faces())
        else:
            mesh.remove_degenerate_faces()
        if hasattr(mesh, 'unique_faces'):
            mesh.update_faces(mesh.unique_faces())
        else:
            mesh.remove_duplicate_faces()
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        if mesh.is_watertight:
            break
    return mesh

def process_and_save(root_dir, subfolders, obj_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    report = []
    for sub in subfolders:
        obj_path = os.path.join(root_dir, sub, obj_name)
        out_path = os.path.join(out_dir, f'{sub}_watertight.obj')
        if os.path.exists(obj_path):
            mesh = try_trimesh_load(obj_path)
            if isinstance(mesh, str):
                # 尝试open3d
                mesh = try_open3d_load(obj_path)
            if isinstance(mesh, str):
                report.append((sub, mesh, False, None))
                continue
            try:
                if not mesh.is_watertight:
                    mesh = recursive_fix(mesh, max_iter=20)
                mesh.export(out_path)
                report.append((sub, 'ok', mesh.is_watertight, out_path))
            except Exception as e:
                report.append((sub, f'fix/export error: {e}', False, None))
        else:
            report.append((sub, 'Not found', False, None))
    return report

if __name__ == '__main__':
    root = '/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask'
    out_dir = '/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_watertight'
    # 自动遍历所有dataXXXX子文件夹
    subfolders = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d.startswith('data')]
    obj_name = 'model.obj'
    results = process_and_save(root, sorted(subfolders), obj_name, out_dir)
    with open('/root/octfusion/mesh_watertight_fix_report.txt', 'w') as f:
        for sub, msg, is_watertight, out_path in results:
            f.write(f'{sub}: {msg}, watertight: {is_watertight}, saved: {out_path}\n')
    print('全部递归修复完成，报告已保存到 /root/octfusion/mesh_watertight_fix_report.txt')
