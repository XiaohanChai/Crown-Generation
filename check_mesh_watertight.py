import trimesh
import os
def is_watertight(obj_path):
    try:
        mesh = trimesh.load(obj_path, force='mesh')
        return mesh.is_watertight
    except Exception as e:
        return f'Error: {e}'

def scan_and_report(root_dir, subfolders, obj_name='model.obj'):
    report = []
    for sub in subfolders:
        obj_path = os.path.join(root_dir, sub, obj_name)
        if os.path.exists(obj_path):
            result = is_watertight(obj_path)
            report.append((sub, result))
        else:
            report.append((sub, 'Not found'))
    return report

if __name__ == '__main__':
    root = '/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask'
    subfolders = [f'data{str(i).zfill(4)}' for i in range(1, 7)]
    results = scan_and_report(root, subfolders)
    with open('/root/octfusion/mesh_watertight_report.txt', 'w') as f:
        for sub, res in results:
            f.write(f'{sub}: {res}\n')
    print('Report saved to /root/octfusion/mesh_watertight_report.txt')
