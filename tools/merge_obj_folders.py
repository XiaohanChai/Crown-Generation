import argparse
from pathlib import Path
import re

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

    shifted_tokens = []
    for token in parts[1:]:
        match = FACE_TOKEN_RE.match(token)
        if not match:
            shifted_tokens.append(token)
            continue

        v_idx, vt_idx, vn_idx = match.group(1), match.group(2), match.group(3)
        new_v = str(int(v_idx) + v_off)

        if '/' not in token:
            shifted_tokens.append(new_v)
            continue

        if vn_idx is None:
            if vt_idx in (None, ''):
                shifted_tokens.append(f'{new_v}/')
            else:
                shifted_tokens.append(f'{new_v}/{int(vt_idx) + vt_off}')
            continue

        new_vt = '' if vt_idx in (None, '') else str(int(vt_idx) + vt_off)
        new_vn = str(int(vn_idx) + vn_off)
        shifted_tokens.append(f'{new_v}/{new_vt}/{new_vn}')

    return 'f ' + ' '.join(shifted_tokens) + '\n'


def merge_obj(file_a, file_b, output_file):
    lines_a = file_a.read_text(encoding='utf-8', errors='ignore').splitlines(keepends=True)
    lines_b = file_b.read_text(encoding='utf-8', errors='ignore').splitlines(keepends=True)

    v_off, vt_off, vn_off = parse_counts(lines_a)

    merged = [f'# merged from {file_a.name} + {file_b.name}\n']
    merged.extend(lines_a)
    for line in lines_b:
        if line.startswith('f '):
            merged.append(shift_face_line(line, v_off, vt_off, vn_off))
        else:
            merged.append(line)

    output_file.write_text(''.join(merged), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Merge same-named OBJ files from two folders.')
    parser.add_argument('--src-a', required=True, help='First source folder')
    parser.add_argument('--src-b', required=True, help='Second source folder')
    parser.add_argument('--dst', required=True, help='Destination folder')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing destination files')
    args = parser.parse_args()

    src_a = Path(args.src_a)
    src_b = Path(args.src_b)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files_a = {p.name: p for p in src_a.glob('*.obj')}
    files_b = {p.name: p for p in src_b.glob('*.obj')}

    common_names = sorted(set(files_a) & set(files_b))
    only_a = sorted(set(files_a) - set(files_b))
    only_b = sorted(set(files_b) - set(files_a))

    created = 0
    skipped = 0
    failed = []

    for name in common_names:
        out_path = dst / name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            merge_obj(files_a[name], files_b[name], out_path)
            created += 1
        except Exception as exc:
            failed.append((name, str(exc)))

    print('summary')
    print(f'src_a={src_a} count={len(files_a)}')
    print(f'src_b={src_b} count={len(files_b)}')
    print(f'common={len(common_names)}')
    print(f'created={created}')
    print(f'skipped={skipped}')
    print(f'failed={len(failed)}')
    print(f'only_a={len(only_a)}')
    print(f'only_b={len(only_b)}')

    if failed:
        print('failed_samples')
        for name, err in failed[:20]:
            print(f'{name}: {err}')


if __name__ == '__main__':
    main()
