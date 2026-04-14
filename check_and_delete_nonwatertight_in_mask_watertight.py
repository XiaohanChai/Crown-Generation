from pathlib import Path
import re
import trimesh

root = Path('/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_watertight')
report = Path('/root/octfusion/mask_watertight_deleted_nonwatertight_report.txt')

pat = re.compile(r'^data\d+_mask_watertight\.obj$')
files = sorted([p for p in root.iterdir() if p.is_file() and pat.match(p.name)])
checked = 0
deleted = []
errors = []

for obj_path in files:
    checked += 1
    try:
        mesh = trimesh.load(obj_path, force='mesh', process=False)
        is_watertight = bool(getattr(mesh, 'is_watertight', False))
        if not is_watertight:
            obj_path.unlink()
            deleted.append(obj_path.name)
    except Exception as exc:
        errors.append((obj_path.name, str(exc)))

lines = [
    f'total_files={len(files)}',
    f'checked={checked}',
    f'deleted_non_watertight={len(deleted)}',
    f'errors={len(errors)}',
    '',
    '[deleted_non_watertight_files]',
]
lines.extend(deleted or ['(none)'])
lines.extend(['', '[errors]'])
if errors:
    for name, err in errors:
        lines.append(f'{name}: {err}')
else:
    lines.append('(none)')

report.write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines[:6]))
print(f'report={report}')
