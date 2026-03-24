#!/usr/bin/env python3
"""
merge_objs.py

合并两个目录中同名子文件夹下的 model.obj。
保留坐标（不进行任何变换），只是把第二个文件的索引偏移后拼接到第一个文件后面。
python merge_objs.py /path/to/folderA /path/to/folderB /path/to/output
"""

import os
import argparse
from pathlib import Path

def parse_obj(path):
    """
    解析 OBJ 文件，返回字典：
    {
      "mtllibs": [lines],
      "verts": [line_without_newline e.g. 'v x y z'],
      "vts": [...],
      "vns": [...],
      "other": [other_lines_preserve],
      "faces": [ list_of_face_records ]
    }
    face record: original tokens parsed to tuples list e.g.
      [ (v_idx, vt_idx_or_None, vn_idx_or_None), ... ]
    Note: indices are converted to integers (positive) relative to file (i.e. negative indices resolved).
    """
    verts = []
    vts = []
    vns = []
    mtllibs = []
    other = []  # lines that are not v/vt/vn/f/mtllib
    faces = []

    # First pass count vertices for negative index handling
    lines = Path(path).read_text(encoding='utf-8', errors='ignore').splitlines()

    # Count numbers for resolving negative indices:
    v_count = sum(1 for l in lines if l.startswith("v "))
    vt_count = sum(1 for l in lines if l.startswith("vt "))
    vn_count = sum(1 for l in lines if l.startswith("vn "))

    for l in lines:
        if l.startswith("v "):
            verts.append(l.strip())
        elif l.startswith("vt "):
            vts.append(l.strip())
        elif l.startswith("vn "):
            vns.append(l.strip())
        elif l.startswith("mtllib "):
            mtllibs.append(l.strip())
        elif l.startswith("f "):
            toks = l.strip().split()[1:]
            face = []
            for tok in toks:
                parts = tok.split('/')
                # parts length 1..3
                # convert to ints, handle empty fields
                def to_int_or_none(s, count):
                    if s == '':
                        return None
                    try:
                        val = int(s)
                    except:
                        return None
                    if val < 0:
                        # negative index: relative to current file's counts
                        return count + 1 + val  # e.g. -1 -> count
                    return val
                v_idx = to_int_or_none(parts[0], v_count) if len(parts) >= 1 else None
                vt_idx = None
                vn_idx = None
                if len(parts) == 2:
                    vt_idx = to_int_or_none(parts[1], vt_count)
                elif len(parts) == 3:
                    vt_idx = to_int_or_none(parts[1], vt_count) if parts[1] != '' else None
                    vn_idx = to_int_or_none(parts[2], vn_count) if parts[2] != '' else None
                face.append((v_idx, vt_idx, vn_idx))
            faces.append(face)
        else:
            other.append(l.rstrip())
    return {
        "mtllibs": mtllibs,
        "verts": verts,
        "vts": vts,
        "vns": vns,
        "other": other,
        "faces": faces
    }

def format_face(face):
    """
    face: list of tuples (v, vt, vn) where entries are ints or None
    return string like: "f v1/vt1/vn1 v2/vt2/vn2 ..."
    """
    parts = []
    for (v, vt, vn) in face:
        if vt is None and vn is None:
            parts.append(str(v))
        elif vt is None and vn is not None:
            # v//vn
            parts.append(f"{v}//{vn}")
        elif vt is not None and vn is None:
            parts.append(f"{v}/{vt}")
        else:
            parts.append(f"{v}/{vt}/{vn}")
    return "f " + " ".join(parts)

def merge_objs(obj1, obj2):
    """
    obj1 and obj2 are parsed dicts from parse_obj.
    Returns a list of strings (lines) for the merged OBJ.
    obj2's indices will be offset appropriately.
    """
    out_lines = []

    # combine mtllibs (unique)
    mtlibs = []
    for m in obj1["mtllibs"] + obj2["mtllibs"]:
        if m not in mtlibs:
            mtlibs.append(m)
    out_lines.extend(mtlibs)

    # vertices
    out_lines.extend(obj1["verts"])
    v_offset = len(obj1["verts"])
    out_lines.extend(obj2["verts"])

    # vts
    out_lines.extend(obj1["vts"])
    vt_offset = len(obj1["vts"])
    out_lines.extend(obj2["vts"])

    # vns
    out_lines.extend(obj1["vns"])
    vn_offset = len(obj1["vns"])
    out_lines.extend(obj2["vns"])

    # include other non-face lines from obj1 (like o/g/usemtl/s), but exclude mtllib lines
    for l in obj1["other"]:
        if not l.startswith("mtllib "):
            out_lines.append(l)
    # write faces from obj1
    for face in obj1["faces"]:
        out_lines.append(format_face(face))

    # optionally separate parts with a comment
    out_lines.append("# ----- merged from second file -----")

    # include other non-face lines from obj2 (but we should update indices in use of negative? those are not index-based)
    for l in obj2["other"]:
        if not l.startswith("mtllib "):
            out_lines.append(l)

    # write faces from obj2 with offsets
    for face in obj2["faces"]:
        new_face = []
        for (v, vt, vn) in face:
            new_v = v + v_offset if v is not None else None
            new_vt = vt + vt_offset if vt is not None else None
            new_vn = vn + vn_offset if vn is not None else None
            new_face.append((new_v, new_vt, new_vn))
        out_lines.append(format_face(new_face))

    return out_lines

def main(folder_a, folder_b, out_folder):
    folder_a = Path(folder_a)
    folder_b = Path(folder_b)
    out_folder = Path(out_folder)

    if not folder_a.is_dir() or not folder_b.is_dir():
        raise SystemExit("folder_a 或 folder_b 路径不存在或不是文件夹。")

    out_folder.mkdir(parents=True, exist_ok=True)

    # 列出两边的子文件夹名称
    subs_a = {p.name for p in folder_a.iterdir() if p.is_dir()}
    subs_b = {p.name for p in folder_b.iterdir() if p.is_dir()}

    common = sorted(list(subs_a & subs_b))
    if not common:
        print("没有同名子文件夹（交集为空）。")
        return

    print(f"找到 {len(common)} 个同名子文件夹，将对它们进行合并：")
    for name in common:
        print(" -", name)
    print()

    for name in common:
        a_model = folder_a / name / "model.obj"
        b_model = folder_b / name / "model.obj"

        if not a_model.is_file():
            print(f"[跳过] {name} 在 folder_a 中缺少 model.obj：{a_model}")
            continue
        if not b_model.is_file():
            print(f"[跳过] {name} 在 folder_b 中缺少 model.obj：{b_model}")
            continue

        print(f"合并 {name} ...")
        obj1 = parse_obj(a_model)
        obj2 = parse_obj(b_model)
        merged_lines = merge_objs(obj1, obj2)

        target_dir = out_folder / name
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / "model.obj"
        out_path.write_text("\n".join(merged_lines) + "\n", encoding='utf-8')
        print(f" 写出 -> {out_path}")

    print("全部完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge model.obj from same-named subfolders of two folders.")
    parser.add_argument("folder_a", help="第一文件夹路径（包含若干子文件夹，每个子文件夹包含 model.obj）")
    parser.add_argument("folder_b", help="第二文件夹路径（包含若干子文件夹，每个子文件夹包含 model.obj）")
    parser.add_argument("out_folder", help="输出根文件夹路径（会创建同名子文件夹并写入 model.obj）")
    args = parser.parse_args()
    main(args.folder_a, args.folder_b, args.out_folder)
