from pathlib import Path
import re

first_root = Path('/root/octfusion/data/mask_crown/crown_585/mesh_obj/basic_files/mask_notwatertight')
second_root = Path('/root/octfusion/data/mask_crown/crown_585/mesh_obj/pair')
out_root = Path('/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_notwatertight_pair')
out_root.mkdir(parents=True, exist_ok=True)

face_token_re = re.compile(r'^(\d+)(?:/(\d*)?(?:/(\d+))?)?$')


def parse_counts(lines):
    num_v = num_vt = num_vn = 0
    for line in lines:
        if line.startswith('v '):
            num_v += 1
        elif line.startswith('vt '):
            num_vt += 1
        elif line.startswith('vn '):
            num_vn += 1
    return num_v, num_vt, num_vn


def shift_face_line(line, v_off, vt_off, vn_off):
    parts = line.strip().split()
    if not parts or parts[0] != 'f':
        return line

    shifted = []
    for token in parts[1:]:
        match = face_token_re.match(token)
        if not match:
            shifted.append(token)
            continue

        v_idx, vt_idx, vn_idx = match.group(1), match.group(2), match.group(3)
        new_v = str(int(v_idx) + v_off)

        if '/' not in token:
            shifted.append(new_v)
            continue

        if vn_idx is None:
            if vt_idx in (None, ''):
                shifted.append(f'{new_v}/')
            else:
                shifted.append(f'{new_v}/{int(vt_idx) + vt_off}')
            continue

        new_vt = '' if vt_idx in (None, '') else str(int(vt_idx) + vt_off)
        new_vn = str(int(vn_idx) + vn_off)
        shifted.append(f'{new_v}/{new_vt}/{new_vn}')

    return 'f ' + ' '.join(shifted) + '\n'


def merge_obj(first_path, second_path, out_path):
    first_lines = first_path.read_text(encoding='utf-8', errors='ignore').splitlines(keepends=True)
    second_lines = second_path.read_text(encoding='utf-8', errors='ignore').splitlines(keepends=True)

    v_off, vt_off, vn_off = parse_counts(first_lines)

    merged_lines = [f'# merged from {first_path} + {second_path}\n']
    merged_lines.extend(first_lines)

    for line in second_lines:
        if line.startswith('f '):
            merged_lines.append(shift_face_line(line, v_off, vt_off, vn_off))
        else:
            merged_lines.append(line)

    out_path.write_text(''.join(merged_lines), encoding='utf-8')


def main():
    first_ids = sorted([p.name for p in first_root.iterdir() if p.is_dir() and p.name.startswith('data')])
    second_ids = {p.name for p in second_root.iterdir() if p.is_dir() and p.name.startswith('data')}

    created = 0
    missing_first = 0
    missing_second = 0
    failed = 0

    for idx, sid in enumerate(first_ids, 1):
        first_obj = first_root / sid / 'model.obj'
        second_obj = second_root / sid / 'model.obj'

        if not first_obj.exists():
            missing_first += 1
            continue
        if sid not in second_ids or not second_obj.exists():
            missing_second += 1
            continue

        out_dir = out_root / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_obj = out_dir / 'model.obj'

        try:
            merge_obj(first_obj, second_obj, out_obj)
            created += 1
            if created % 100 == 0:
                print(f'created={created} processed={idx}/{len(first_ids)}', flush=True)
        except Exception:
            failed += 1

    print('summary')
    print(f'first_ids={len(first_ids)}')
    print(f'created={created}')
    print(f'missing_first={missing_first}')
    print(f'missing_second={missing_second}')
    print(f'failed={failed}')


if __name__ == '__main__':
    main()
