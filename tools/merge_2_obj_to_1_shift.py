import os
import re

# ========== 配置 ========== (按需修改)
folder1 = "/root/octfusion/data/mask_crown/bbox_object_crown"
folder2 = "/root/octfusion/data/mask_crown/object_crown"
output_folder = "/root/octfusion/data/mask_crown/mask_object_crown"
translate_z_for_folder2 = 0

# ========== 正则与工具函数（与之前逻辑一致） ==========
_vertex_re = re.compile(r'^(\s*)v\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)(.*)$')

def count_obj_elements(lines):
    n_v = n_vt = n_vn = 0
    for ln in lines:
        if ln.startswith('v '):
            n_v += 1
        elif ln.startswith('vt '):
            n_vt += 1
        elif ln.startswith('vn '):
            n_vn += 1
    return n_v, n_vt, n_vn

def adjust_index(idx_str, offset, base_count):
    if idx_str == '' or idx_str is None:
        return ''
    try:
        idx = int(idx_str)
    except ValueError:
        return idx_str
    if idx > 0:
        abs_idx = idx
    else:
        abs_idx = base_count + 1 + idx
    new_idx = offset + abs_idx
    return str(new_idx)

def adjust_face_token(token, v_offset, vt_offset, vn_offset, base_counts):
    parts = token.split('/')
    while len(parts) < 3:
        parts.append('')
    v_part, vt_part, vn_part = parts[0], parts[1], parts[2]
    v_new = adjust_index(v_part, v_offset, base_counts['v']) if v_part != '' else ''
    vt_new = adjust_index(vt_part, vt_offset, base_counts['vt']) if vt_part != '' else ''
    vn_new = adjust_index(vn_part, vn_offset, base_counts['vn']) if vn_part != '' else ''
    if vt_part == '' and vn_part != '':
        return f"{v_new}//{vn_new}"
    if vt_part != '' and vn_part == '':
        return f"{v_new}/{vt_new}"
    if vt_part == '' and vn_part == '':
        return f"{v_new}"
    return f"{v_new}/{vt_new}/{vn_new}"

def translate_vertices_in_lines(lines, dz):
    out_lines = []
    for ln in lines:
        m = _vertex_re.match(ln)
        if m:
            leading_ws = m.group(1)
            xs = m.group(2)
            ys = m.group(3)
            zs = m.group(4)
            tail = m.group(5)
            try:
                x = float(xs)
                y = float(ys)
                z = float(zs) + dz
                new_v_line = f"{leading_ws}v {x:.6f} {y:.6f} {z:.6f}{tail}\n"
            except Exception:
                new_v_line = ln
            out_lines.append(new_v_line)
        else:
            out_lines.append(ln)
    return out_lines

def merge_two_objs(path1, path2, output_path, dz_for_2=0.0):
    with open(path1, 'r', encoding='utf-8', errors='ignore') as f:
        lines1 = f.readlines()
    with open(path2, 'r', encoding='utf-8', errors='ignore') as f:
        raw_lines2 = f.readlines()

    lines2 = translate_vertices_in_lines(raw_lines2, dz_for_2)

    n_v1, n_vt1, n_vn1 = count_obj_elements(lines1)
    n_v2, n_vt2, n_vn2 = count_obj_elements(lines2)

    v_offset = n_v1
    vt_offset = n_vt1
    vn_offset = n_vn1
    base_counts = {'v': n_v2, 'vt': n_vt2, 'vn': n_vn2}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as out:
        out.writelines(lines1)
        if not lines1 or not lines1[-1].endswith('\n'):
            out.write('\n')
        for ln in lines2:
            if ln.lstrip().startswith('f '):
                parts = ln.strip().split()
                new_tokens = ['f']
                for tok in parts[1:]:
                    new_tokens.append(adjust_face_token(tok, v_offset, vt_offset, vn_offset, base_counts))
                out.write(' '.join(new_tokens) + '\n')
            else:
                out.write(ln)

# ========== 路径映射构建 ==========
def build_relpath_map(root):
    relmap = {}
    root = os.path.abspath(root)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith('.obj'):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            rel_posix = rel.replace(os.path.sep, '/')
            relmap[rel_posix] = full
    return relmap

# ========== 新的匹配逻辑（先尝试用完整相对路径/父路径匹配） ==========
def find_best_match_for_relpath(rel1, map2_keys):
    """
    优先匹配顺序：
      0) 完整相对路径相等（例如 rel1 == "0001/data0001/model.obj"）
      1) 父路径 + "model.obj"（例如 rel1 父目录是 "0001/data0001"，则尝试 "0001/data0001/model.obj"）
      2) 然后退回到按文件名/目录名的匹配（原先的 find_best_match_for_name 行为）
    返回 map2 的 key 或 None。
    """
    # 0) exact relative path match
    if rel1 in map2_keys:
        return rel1

    # 1) try parent_path + /model.obj
    parent = os.path.dirname(rel1)  # posix style because rel1 使用了 '/'
    if parent:
        cand = parent.rstrip('/') + '/model.obj'
        if cand in map2_keys:
            return cand

    # 2) fallback to name-based matching (原始逻辑)
    name = os.path.splitext(os.path.basename(rel1))[0]
    # reuse original name-based priority:
    # a) */<name>/model.obj
    suffix1 = f"/{name}/model.obj"
    for k in map2_keys:
        if k.endswith(suffix1):
            return k
    # b) */<name>.obj
    suffix2 = f"/{name}.obj"
    for k in map2_keys:
        if k.endswith(suffix2):
            return k
    # c) */<name>/*.obj
    mid = f"/{name}/"
    for k in map2_keys:
        if mid in k and k.endswith('.obj'):
            return k
    # d) basename == <name>.obj
    base = f"{name}.obj"
    for k in map2_keys:
        if os.path.basename(k) == base:
            return k
    return None

def merge_obj_folders_mixed(folder1_root, folder2_root, output_root, dz_for_folder2=0.0):
    os.makedirs(output_root, exist_ok=True)
    map1 = build_relpath_map(folder1_root)
    map2 = build_relpath_map(folder2_root)

    outputs = []
    unmatched1 = []
    used_map2 = set()

    for rel1, abs1 in sorted(map1.items()):
        # rel1 例如 "0001/data0001/model.obj"
        match_rel2 = find_best_match_for_relpath(rel1, map2.keys())
        if match_rel2:
            abs2 = map2[match_rel2]
            used_map2.add(match_rel2)
            outpath = os.path.join(output_root, rel1.replace('/', os.path.sep))
            print(f"匹配成功:\n  folder1: {abs1}\n  folder2: {abs2}\n→ {outpath}")
            merge_two_objs(abs1, abs2, outpath, dz_for_2=dz_for_folder2)
            outputs.append(outpath)
        else:
            unmatched1.append(rel1)

    unused2 = sorted(set(map2.keys()) - used_map2)

    if unmatched1:
        print("\n⚠ 以下 folder1 文件未找到对应的 folder2 匹配（示例最多 20）：")
        for r in unmatched1[:20]:
            print("  " + r)
    if unused2:
        print("\n⚠ 以下 folder2 文件未被匹配（示例最多 20）：")
        for r in unused2[:20]:
            print("  " + r)
    return outputs

# ========== 主入口 ==========
if __name__ == '__main__':
    outs = merge_obj_folders_mixed(folder1, folder2, output_folder, dz_for_folder2=translate_z_for_folder2)
    if outs:
        print(f"\n✅ 合并完成，生成 {len(outs)} 个文件（示例最多 20 条）：")
        for p in outs[:20]:
            print(p)
    else:
        print("\n❌ 未生成任何文件")
