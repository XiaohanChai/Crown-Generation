import trimesh
import os

def analyze_mesh(mesh):
    info = {}
    info['is_watertight'] = mesh.is_watertight
    info['num_open_edges'] = len(mesh.edges_boundary) if hasattr(mesh, 'edges_boundary') else 'N/A'
    info['open_edges'] = mesh.edges_boundary.tolist() if hasattr(mesh, 'edges_boundary') else []
    if hasattr(mesh, 'faces_sparse') and mesh.faces_sparse is not None:
        try:
            info['num_bad_faces'] = mesh.faces_sparse.getnnz() if hasattr(mesh.faces_sparse, 'getnnz') else mesh.faces_sparse.shape[0]
        except Exception:
            info['num_bad_faces'] = 'error'
    else:
        info['num_bad_faces'] = 'N/A'
    try:
        if hasattr(mesh, 'face_normals') and hasattr(mesh, 'vertex_normals'):
            # 计算每个面的法线与该面顶点法线平均值的夹角
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
    info['non_manifold_edges'] = mesh.edges_non_manifold.tolist() if hasattr(mesh, 'edges_non_manifold') else []
    return info

if __name__ == '__main__':
    obj_path = '/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_watertight/data0005_watertight.obj'
    report_path = '/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_watertight/data0005_analysis.txt'
    mesh = trimesh.load(obj_path, force='mesh', process=True)
    info = analyze_mesh(mesh)
    with open(report_path, 'w') as f:
        for k, v in info.items():
            f.write(f'{k}: {v}\n')
    print(f'分析报告已保存到 {report_path}')
