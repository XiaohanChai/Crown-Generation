import os
bad_ids = [
    'data0005','data0012','data0025','data0068','data0114','data0190','data0195','data0225','data0270','data0288','data0374','data0405','data0531','data0532','data0567'
]
base = '/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_watertight'
for bid in bad_ids:
    f = os.path.join(base, f'{bid}_watertight.obj')
    if os.path.exists(f):
        os.remove(f)
        print(f'Removed: {f}')
    else:
        print(f'Not found: {f}')
