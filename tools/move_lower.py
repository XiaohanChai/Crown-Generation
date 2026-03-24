#!/usr/bin/env python3
"""
move_lower.py

将 lower_crown 下所有 */model.obj 的顶点沿本地 z 轴平移指定偏移（默认 -5），
并把结果输出到 lower_crown-<offset>/.../model.obj（目录结构保持不变）。

用法:
    python move_lower.py
    python move_lower.py --src /root/octfusion/data/mask_crown/lower_crown --offset -5 --dst-suffix -5
    python move_lower.py --src /root/octfusion/data/mask_crown/ --offset -2.5 --dst-suffix -2p5

默认行为：src="/root/octfusion/data/mask_crown/lower_crown", offset=-5, dst_suffix="-5"
"""

import argparse
from pathlib import Path
import shutil
import sys

def translate_obj_file(src_path: Path, dst_path: Path, z_offset: float):
    """
    逐行读取 .obj 文件，只处理以 'v ' 开头的顶点行，增加 z_offset 到第三个数值上。
    其他行原样写入。
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open('r', encoding='utf-8', errors='surrogateescape') as fin, \
         dst_path.open('w', encoding='utf-8', errors='surrogateescape') as fout:
        for line in fin:
            if line.startswith('v '):
                # split 并保持可能的额外空格风格较难精确还原，采用常见格式输出
                parts = line.strip().split()
                # parts[0] == 'v'
                if len(parts) >= 4:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        z += z_offset
                        # 若有更多数值（例如 w），保留原样
                        rest = parts[4:]
                        # 写回，使用一般的浮点格式（保留 6 位小数以保持可读性）
                        new_line = 'v {:.6f} {:.6f} {:.6f}'.format(x, y, z)
                        if rest:
                            new_line += ' ' + ' '.join(rest)
                        fout.write(new_line + '\n')
                    except ValueError:
                        # 如果解析失败，退回写原行（避免损坏文件）
                        fout.write(line)
                else:
                    # 顶点行但不符合期望格式，直接写回
                    fout.write(line)
            else:
                # 其他行（vn, vt, f, o, g, comments 等）原样写回
                fout.write(line)

def main():
    parser = argparse.ArgumentParser(description="Translate .obj vertices along local z and write to mirrored output tree.")
    parser.add_argument('--src', type=str, default='/root/octfusion/data/mask_crown/lower_crown', help='源目录（默认 lower_crown）')
    parser.add_argument('--offset', type=float, default=-3.0, help='沿 z 轴的平移量（默认 -3）')
    parser.add_argument('--dst-suffix', type=str, default='-3', help='输出目录后缀，例如 "-3" 会把 lower_crown -> lower_crown-3')
    parser.add_argument('--pattern', type=str, default='**/model.obj', help='查找模式（glob），默认 **/model.obj')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的输出文件（默认是覆盖）')
    args = parser.parse_args()

    src_root = Path(args.src)
    if not src_root.exists():
        print(f"错误：源目录不存在：{src_root}", file=sys.stderr)
        sys.exit(1)
    dst_root = src_root.parent / (src_root.name + args.dst_suffix)

    files = list(src_root.glob(args.pattern))
    if not files:
        print("未找到任何匹配的 obj 文件。模式：", args.pattern)
        sys.exit(0)

    print(f"找到 {len(files)} 个文件，正在处理：")
    for f in files:
        rel = f.relative_to(src_root)
        dst = dst_root / rel
        print(f"  {f} -> {dst}")
        # 直接写入（会覆盖），若想要备份可在此处添加备份逻辑
        translate_obj_file(f, dst, args.offset)

    print("完成。输出目录：", dst_root)

if __name__ == '__main__':
    main()
