import trimesh
import os

def analyze_mesh(mesh):
    info = {}
    info['is_watertight'] = mesh.is_watertight
    info['num_open_edges'] = len(mesh.edges_boundary) if hasattr(mesh, 'edges_boundary') else 'N/A'
    info['num_bad_faces'] = mesh.faces_sparse.getnnz() if hasattr(mesh, 'faces_sparse') and mesh.faces_sparse is not None and hasattr(mesh.faces_sparse, 'getnnz') else 'N/A'
    try:
        if hasattr(mesh, 'face_normals') and hasattr(mesh, 'vertex_normals'):
            vnorm = mesh.vertex_normals[mesh.faces].mean(axis=1)
            dots = (mesh.face_normals * vnorm).sum(axis=1)
            info['num_inverted_faces'] = int((dots < 0).sum())
        else:
            info['num_inverted_faces'] = 'N/A'
    except Exception:
        info['num_inverted_faces'] = 'error'
    info['num_self_intersections'] = len(mesh.self_intersecting_faces) if hasattr(mesh, 'self_intersecting_faces') else 'N/A'
    info['num_duplicate_vertices'] = mesh.vertices.shape[0] - trimesh.grouping.unique_rows(mesh.vertices)[0].shape[0]
    info['is_winding_consistent'] = mesh.is_winding_consistent if hasattr(mesh, 'is_winding_consistent') else 'N/A'
    info['euler_number'] = mesh.euler_number if hasattr(mesh, 'euler_number') else 'N/A'
    info['non_manifold_edges'] = len(mesh.edges_non_manifold) if hasattr(mesh, 'edges_non_manifold') else 'N/A'
    return info

if __name__ == '__main__':
    base = '/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_watertight'
    ids = ['data0005','data0012','data0025','data0068','data0114','data0190','data0195','data0225','data0270','data0288','data0374','data0405','data0531','data0532','data0567']
    with open('/root/octfusion/mesh_watertight_bad_analysis.txt','w') as f:
        for sid in ids:
            obj_path = os.path.join(base, f'{sid}_watertight.obj')
            try:
                mesh = trimesh.load(obj_path, force='mesh', process=True)
                info = analyze_mesh(mesh)
                f.write(f'{sid}: {info}\n')
            except Exception as e:
                f.write(f'{sid}: error: {e}\n')
    print('不水密模型分析报告已生成')
