import os

def merge_obj_files(obj1_path, obj2_path, out_path):
    with open(obj1_path, 'r') as f1, open(obj2_path, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
    # 合并顶点和面，注意面索引偏移
    v1 = [l for l in lines1 if l.startswith('v ')]
    f1s = [l for l in lines1 if l.startswith('f ')]
    v2 = [l for l in lines2 if l.startswith('v ')]
    f2s = [l for l in lines2 if l.startswith('f ')]
    # 计算顶点偏移
    v1_count = len(v1)
    # 修正pair的面索引
    def shift_face(line, offset):
        parts = line.strip().split()
        new_parts = [parts[0]]
        for p in parts[1:]:
            idx = p.split('/')
            idx[0] = str(int(idx[0]) + offset)
            new_parts.append('/'.join(idx))
        return ' '.join(new_parts) + '\n'
    f2s_shifted = [shift_face(l, v1_count) for l in f2s]
    # 合并
    with open(out_path, 'w') as f:
        for l in v1: f.write(l)
        for l in v2: f.write(l)
        for l in f1s: f.write(l)
        for l in f2s_shifted: f.write(l)

mask_dir = '/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_watertight'
pair_dir = '/root/octfusion/data/mask_crown/crown_585/mesh_obj/pair'
out_dir = '/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_pair'
os.makedirs(out_dir, exist_ok=True)

mask_subs = set([d for d in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, d))])
pair_subs = set([d for d in os.listdir(pair_dir) if os.path.isdir(os.path.join(pair_dir, d))])
common_subs = sorted(mask_subs & pair_subs)

for sub in common_subs:
    mask_obj = os.path.join(mask_dir, sub, 'model.obj')
    pair_obj = os.path.join(pair_dir, sub, 'model.obj')
    if os.path.exists(mask_obj) and os.path.exists(pair_obj):
        out_sub = os.path.join(out_dir, sub)
        os.makedirs(out_sub, exist_ok=True)
        out_obj = os.path.join(out_sub, 'model.obj')
        merge_obj_files(mask_obj, pair_obj, out_obj)
        print(f'Merged: {sub}/model.obj')
    else:
        print(f'Skipped: {sub} (missing model.obj)')
