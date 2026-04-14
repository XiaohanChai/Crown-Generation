import os
import shutil

src_dir = '/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_watertight'
obj_files = [f for f in os.listdir(src_dir) if f.endswith('_watertight.obj')]

for obj_file in obj_files:
    # 去掉_watertight后缀，得到dataXXXX
    base_name = obj_file.replace('_watertight.obj', '')
    target_dir = os.path.join(src_dir, base_name)
    os.makedirs(target_dir, exist_ok=True)
    src_path = os.path.join(src_dir, obj_file)
    dst_path = os.path.join(target_dir, 'model.obj')
    shutil.move(src_path, dst_path)
    print(f'Moved {src_path} -> {dst_path}')
