import os

base_dir = '/root/octfusion/data/mask_crown_750/filelist'

# 三个输入文件
file_mask   = os.path.join(base_dir, 'mask_pair_crown.txt')
file_bbox   = os.path.join(base_dir, 'bbox_crown.txt')
file_pair   = os.path.join(base_dir, 'pair_crown.txt')

# 输出文件
output_file = os.path.join(base_dir, 'mask_crown_750.txt')

# 读取并排序
def read_and_sort(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    # 按编号排序，比如 data0001 -> 1，data0010 -> 10
    lines.sort(key=lambda x: int(x.split('/')[-1].replace('data','')))
    return lines

mask_list = read_and_sort(file_mask)
bbox_list = read_and_sort(file_bbox)
pair_list = read_and_sort(file_pair)

# 确保长度一致，否则可能有缺失
min_len = min(len(mask_list), len(bbox_list), len(pair_list))

with open(output_file, 'w', encoding='utf-8') as f:
    for i in range(min_len):
        f.write(mask_list[i] + '\n')
        f.write(bbox_list[i] + '\n')
        f.write(pair_list[i] + '\n')

print("✅ 已生成:", output_file)
