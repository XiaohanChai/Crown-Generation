import shutil
src = '/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask/data0005/model.obj'
dst = '/root/octfusion/data/mask_crown/crown_585/mesh_obj/mask_watertight/data0005_watertight.obj'
shutil.copyfile(src, dst)
print('data0005_watertight.obj 已重置为原始模型，准备递归修复')
