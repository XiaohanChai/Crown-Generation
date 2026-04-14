from pathlib import Path
import argparse
import re

BASE = Path('/root/octfusion/data/mask_crown/crown_585/mesh_obj')

FACE_TOKEN_RE = re.compile(r'^(\d+)(?:/(\d*)?(?:/(\d+))?)?$')


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
        match = FACE_TOKEN_RE.match(token)
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


def merge_obj(first_path, second_path, output_path):
    first_lines = first_path.read_text(encoding='utf-8', errors='ignore').splitlines(keepends=True)
    second_lines = second_path.read_text(encoding='utf-8', errors='ignore').splitlines(keepends=True)

    v_off, vt_off, vn_off = parse_counts(first_lines)

    merged_lines = [f'# merged from {first_path.name} + {second_path.name}\n']
    merged_lines.extend(first_lines)

    for line in second_lines:
        if line.startswith('f '):
            merged_lines.append(shift_face_line(line, v_off, vt_off, vn_off))
        else:
            merged_lines.append(line)

    output_path.write_text(''.join(merged_lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Merge same-name OBJ files from two folders.')
    parser.add_argument('--first', type=str, default=str(BASE / 'upper'),
                        help='First input folder containing OBJ files.')
    parser.add_argument('--second', type=str, default=str(BASE / 'lower'),
                        help='Second input folder containing OBJ files.')
    parser.add_argument('--out', type=str, default=str(BASE / 'pair'),
                        help='Output folder to write merged OBJ files.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing outputs instead of skipping.')
    args = parser.parse_args()

    first_dir = Path(args.first)
    second_dir = Path(args.second)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    first_files = {p.name: p for p in first_dir.glob('*.obj')}
    second_files = {p.name: p for p in second_dir.glob('*.obj')}

    common_names = sorted(set(first_files) & set(second_files))
    only_first = sorted(set(first_files) - set(second_files))
    only_second = sorted(set(second_files) - set(first_files))

    created = 0
    skipped = 0
    failed = []

    for idx, name in enumerate(common_names, 1):
        output_path = out_dir / name
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            merge_obj(first_files[name], second_files[name], output_path)
            created += 1
            if created % 20 == 0:
                print(f'created_new={created} (processed={idx}/{len(common_names)})', flush=True)
        except Exception as exc:
            failed.append((name, str(exc)))

    print('summary', flush=True)
    print(f'first={len(first_files)}', flush=True)
    print(f'second={len(second_files)}', flush=True)
    print(f'common={len(common_names)}', flush=True)
    print(f'skipped_existing={skipped}', flush=True)
    print(f'created_new={created}', flush=True)
    print(f'failed={len(failed)}', flush=True)
    print(f'only_first={len(only_first)}', flush=True)
    print(f'only_second={len(only_second)}', flush=True)

    if failed:
        print('failed_samples', flush=True)
        for name, err in failed[:20]:
            print(f'{name}: {err}', flush=True)


if __name__ == '__main__':
    main()
