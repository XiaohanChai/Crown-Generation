from pathlib import Path
import re

mask_root = Path('/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_watertight')
pair_root = Path('/root/octfusion/data/mask_crown/crown_585/mesh_obj/pair')
out_root = Path('/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_watertight_pair')
out_root.mkdir(parents=True, exist_ok=True)

face_token_re = re.compile(r'^(\d+)(?:/(\d*)?(?:/(\d+))?)?$')


def parse_counts(lines):
    nv = nvt = nvn = 0
    for line in lines:
        if line.startswith('v '):
            nv += 1
        elif line.startswith('vt '):
            nvt += 1
        elif line.startswith('vn '):
            nvn += 1
    return nv, nvt, nvn


def shift_face_line(line, v_off, vt_off, vn_off):
    parts = line.strip().split()
    if not parts or parts[0] != 'f':
        return line

    shifted = []
    for token in parts[1:]:
        m = face_token_re.match(token)
        if not m:
            shifted.append(token)
            continue

        v_idx, vt_idx, vn_idx = m.group(1), m.group(2), m.group(3)
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


def merge_obj(mask_obj: Path, pair_obj: Path, out_obj: Path):
    a = mask_obj.read_text(encoding='utf-8', errors='ignore').splitlines(keepends=True)
    b = pair_obj.read_text(encoding='utf-8', errors='ignore').splitlines(keepends=True)

    v_off, vt_off, vn_off = parse_counts(a)

    merged = [f'# merged from {mask_obj} + {pair_obj}\n']
    merged.extend(a)

    for line in b:
        if line.startswith('f '):
            merged.append(shift_face_line(line, v_off, vt_off, vn_off))
        else:
            merged.append(line)

    out_obj.write_text(''.join(merged), encoding='utf-8')


def main():
    mask_ids = sorted([p.name for p in mask_root.iterdir() if p.is_dir() and p.name.startswith('data')])
    pair_ids = {p.name for p in pair_root.iterdir() if p.is_dir() and p.name.startswith('data')}

    created = 0
    missing_mask = 0
    missing_pair = 0
    failed = 0

    for idx, did in enumerate(mask_ids, 1):
        mask_obj = mask_root / did / 'model.obj'
        pair_obj = pair_root / did / 'model.obj'
        out_dir = out_root / did
        out_obj = out_dir / 'model.obj'

        if not mask_obj.exists():
            missing_mask += 1
            continue
        if did not in pair_ids or not pair_obj.exists():
            missing_pair += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            merge_obj(mask_obj, pair_obj, out_obj)
            created += 1
            if created % 50 == 0:
                print(f'created={created} processed={idx}/{len(mask_ids)}', flush=True)
        except Exception:
            failed += 1

    print('summary')
    print(f'mask_ids={len(mask_ids)}')
    print(f'created={created}')
    print(f'missing_mask={missing_mask}')
    print(f'missing_pair={missing_pair}')
    print(f'failed={failed}')


if __name__ == '__main__':
    main()
