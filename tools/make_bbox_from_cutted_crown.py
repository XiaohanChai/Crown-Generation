# !/usr/bin/env python3
"""
make_bbox_batch_obj.py
交互式批量为文件夹中每个 .obj 网格生成包围盒立方体并导出为 .obj 文件。
"""

import sys
import traceback
from pathlib import Path

try:
    import pymeshlab
except Exception:
    print("错误：未安装 pymeshlab，请先执行：")
    print("    pip install pymeshlab")
    sys.exit(1)


# 只处理 OBJ 文件
SUPPORTED_EXTS = {'.obj'}


def make_bbox(input_path: Path, output_path: Path) -> None:
    """
    为单个输入 OBJ 模型生成轴对齐包围盒立方体并保存为 OBJ。
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_path))
    m = ms.current_mesh()

    # 计算包围盒
    bb = m.bounding_box()
    cx, cy, cz = bb.center().tolist()
    dx, dy, dz = bb.dim_x(), bb.dim_y(), bb.dim_z()

    # 创建单位立方体
    ms.create_cube()

    # 缩放并平移到包围盒位置
    ms.compute_matrix_from_translation_rotation_scale(
        scalex=dx, scaley=dy, scalez=dz,
        translationx=cx, translationy=cy, translationz=cz,
        rotationx=0.0, rotationy=0.0, rotationz=0.0
    )

    # 保存为 OBJ
    ms.save_current_mesh(str(output_path))
    print(f"    ✅ 已生成: {output_path.name}")


def batch_process_folder(input_dir: Path, output_dir: Path, recursive: bool = False) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"输入文件夹不存在或不是目录: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有 .obj 文件
    files = (
        [p for p in input_dir.rglob('*.obj')] if recursive
        else [p for p in input_dir.glob('*.obj')]
    )

    if not files:
        print("⚠️ 未找到任何 .obj 文件。")
        return

    print(f"发现 {len(files)} 个 .obj 文件，开始处理...\n")
    errors = []

    for idx, f in enumerate(files, 1):
        try:
            print(f"[{idx}/{len(files)}] 处理: {f.name}")
            out_name = f.stem + ".obj"
            out_path = output_dir / out_name

            if f.resolve() == out_path.resolve():
                print("    ⚠️ 跳过：输入与输出路径相同，避免覆盖。")
                continue

            make_bbox(f, out_path)

        except Exception as e:
            errors.append((f.name, str(e)))
            print(f"    ❌ 错误: {e}")
            traceback.print_exc()

    print("\n=== 批处理完成 ===")
    print(f"成功: {len(files) - len(errors)}，失败: {len(errors)}")
    if errors:
        print("失败文件列表：")
        for fname, err in errors:
            print(f" - {fname}: {err}")


def prompt_path(prompt_text: str) -> Path:
    while True:
        s = input(prompt_text).strip()
        if not s:
            print("请输入有效路径。")
            continue
        return Path(s).expanduser()


def main():
    print("=== PyMeshLab 批量生成 OBJ 包围盒立方体 ===")
    in_dir = prompt_path("请输入输入文件夹路径（含 .obj 模型）： ")
    out_dir = prompt_path("请输入输出文件夹路径（保存生成的包围盒 .obj）： ")
    rec = input("是否递归处理子文件夹？(y/N)： ").strip().lower() == 'y'

    batch_process_folder(in_dir, out_dir, recursive=rec)


if __name__ == '__main__':
    main()
