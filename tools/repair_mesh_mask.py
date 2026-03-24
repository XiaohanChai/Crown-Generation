# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import time
import wget
import shutil
import torch
import ocnn
import trimesh
import logging
import mesh2sdf
import zipfile
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from plyfile import PlyData, PlyElement

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default="convert_mesh_to_sdf")
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=-1)
parser.add_argument('--sdf_size', type=int, default=128)
args = parser.parse_args()

size = args.sdf_size        # resolution of SDF
level = 2.0 / size            # 2/128 = 0.015625
shape_scale = 0.5    # rescale the shape into [-0.5, 0.5]
project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(project_folder, 'data/mask_crown_750/')
file_folder = 'data/mask_crown_750/'


def create_flag_file(filename):
    r''' Creates a flag file to indicate whether some time-consuming works have been done.
    '''
    folder = os.path.dirname(filename)
    os.makedirs(folder, exist_ok=True)

    tmp = filename + '.tmp'
    try:
        with open(tmp, 'w') as fid:
            fid.write('succ @ ' + time.ctime())
            fid.flush()
            try:
                os.fsync(fid.fileno())
            except Exception:
                # fsync may not be available on some filesystems; ignore if it fails
                pass
        # atomic replace
        os.replace(tmp, filename)
    except Exception:
        # cleanup temp file on failure
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
        raise

def check_folder(filenames: list):
    r''' Checks whether the folder contains the filename exists.
    '''

    for filename in filenames:
        folder = os.path.dirname(filename)
        if not os.path.exists(folder):
            os.makedirs(folder)

def get_filenames(filelist):
    r''' Gets filenames from a filelist.
    '''

    filelist = os.path.join(root_folder, 'filelist', filelist)
    with open(filelist, 'r') as fid:
        lines = fid.readlines()
    filenames = [line.split()[0] for line in lines]
    return filenames

def ply2obj():
    ply_folder = '/root/octfusion/data/mask_crown_350/lower_crown_0'
    obj_folder = '/root/octfusion/data/mask_crown_350/lower_crown'
    os.makedirs(obj_folder, exist_ok=True)

    for fname in os.listdir(ply_folder):
        if fname.endswith('.ply'):
            mesh = trimesh.load(os.path.join(ply_folder, fname))
            obj_name = os.path.splitext(fname)[0] + '.obj'
            mesh.export(os.path.join(obj_folder, obj_name))

# def prepare_original_mesh():
# #     move_obj_to_folder()
# #     regroup_obj_in_mesh()
# #     rename_obj_to_model()

# # def move_obj_to_folder():
# #     r''' Moves .obj files to folders named by the 5th to 8th characters of the filename.
# #     '''
#     mesh_object = '/root/octfusion/data/mask_crown/bbox_crown'
#     for fname in os.listdir(mesh_object):
#         if fname.endswith('.obj'):
#             folder_name = fname[4:8]  # 文件名第5-8位
#             dst_folder = os.path.join(mesh_object, folder_name)
#             os.makedirs(dst_folder, exist_ok=True)
#             src_path = os.path.join(mesh_object, fname)
#             dst_path = os.path.join(dst_folder, fname)
#             shutil.move(src_path, dst_path)

# # def regroup_obj_in_mesh():
# #     '''
# #     遍历 mesh_Object 下的所有子文件夹，在每个子文件夹下新建一个以该文件夹内 .obj 文件名为名的文件夹，
# #     并将该 .obj 文件移动到新建的文件夹中。
# #     '''

# #     mesh_object = '/root/octfusion/data/mask_crown/mask_pair_crown'
#     for folder in os.listdir(mesh_object):
#         folder_path = os.path.join(mesh_object, folder)
#         if os.path.isdir(folder_path):
#             for fname in os.listdir(folder_path):
#                 if fname.endswith('.obj'):
#                     new_folder = os.path.join(folder_path, os.path.splitext(fname)[0])
#                     os.makedirs(new_folder, exist_ok=True)
#                     src_path = os.path.join(folder_path, fname)
#                     dst_path = os.path.join(new_folder, fname)
#                     shutil.move(src_path, dst_path)
#                     print(f"已移动: {src_path} -> {dst_path}")

# # def rename_obj_to_model():
# #     '''
# #     遍历 mesh_Object 下的所有子文件夹，将每个子文件夹中的所有 .obj 文件重命名为 model.obj
# #     '''
# #     mesh_object = '/root/octfusion/data/mask_crown/mask_pair_crown'
#     for folder in os.listdir(mesh_object):
#         folder_path = os.path.join(mesh_object, folder)
#         if os.path.isdir(folder_path):
#             for subfolder in os.listdir(folder_path):
#                 subfolder_path = os.path.join(folder_path, subfolder)
#                 if os.path.isdir(subfolder_path):
#                     for fname in os.listdir(subfolder_path):
#                         if fname.endswith('.obj') and fname != 'model.obj':
#                             src = os.path.join(subfolder_path, fname)
#                             dst = os.path.join(subfolder_path, 'model.obj')
#                             os.rename(src, dst)
#                             print(f"重命名: {src} -> {dst}")

def create_right_folderpath():
    import os
    import shutil

    folder = "/root/octfusion/data/mask_crown_350/mesh_obj/bbox_crown"  

    for filename in os.listdir(folder):
        if filename.startswith("data") and filename.endswith(".obj"):
            name_without_ext = filename[:-4]  
            target_dir = os.path.join(folder, name_without_ext)
            os.makedirs(target_dir, exist_ok=True)
            src_path = os.path.join(folder, filename)
            dst_path = os.path.join(target_dir, "model.obj")
            shutil.move(src_path, dst_path)

    print("完成！")

import os

def generate_mesh_paths():
    base_dir = '/root/octfusion/data/mask_crown_750/mesh_obj/'
    folder = input("请输入要处理的文件夹名称 (例如 bbox_crown)：").strip()

    root_dir = os.path.join(base_dir, folder)
    output_file = os.path.join(base_dir, 'filelist', f'{folder}.txt')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    folders_written = set()

    with open(output_file, 'w', encoding='utf-8') as f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # ✅ 排序子目录和文件名，确保顺序正确
            dirnames.sort()
            filenames.sort()

            if 'model.obj' in filenames:
                relative_dir = os.path.relpath(dirpath, root_dir)
                line = os.path.join(folder, relative_dir)
                if line not in folders_written:
                    f.write(line + '\n')
                    folders_written.add(line)

    print(f"已生成文件: {output_file}")

def convert_mesh_to_sdf():
    run_mesh2sdf()
    # run_mesh2sdf_mp()

def run_mesh2sdf_split():
    r''' Converts the meshes from crown/oppose to SDFs and manifold meshes.
    '''

    print('-> Run mesh2sdf.')
    mesh_scale = 0.8
    filenames = get_filenames('mask_crown_350_2.txt')
    GROUP_SIZE = 3

    total_groups = (len(filenames) + GROUP_SIZE - 1) // GROUP_SIZE
    start_group = max(0, args.start)
    end_group = total_groups if args.end < 0 else min(args.end, total_groups)

    if start_group >= end_group:
        print('No meshes to process (invalid start/end groups).')
        return

    start_idx = start_group * GROUP_SIZE
    end_idx = min(len(filenames), end_group * GROUP_SIZE)
    print(f'Processing filenames[{start_idx}:{end_idx}) (groups {start_group} to {end_group - 1}).')

    center_merge = None
    scale_merge = None

    for i in range(start_idx, end_idx):
        filename = filenames[i]
        # filename = "pair_crown/data0001"
        # filename_raw = os.path.join(
                # file_folder, 'mesh_Object', filename, 'model.obj')
        filename_raw = os.path.join(
                file_folder, 'mesh_obj', filename, 'model.obj')
        filename_obj = os.path.join(root_folder, 'mesh', filename + '.obj')
        filename_box = os.path.join(root_folder, 'bbox', filename + '.npz')
        filename_npy = os.path.join(root_folder, 'sdf', filename + '.npy')
        check_folder([filename_obj, filename_box, filename_npy])
        # if os.path.exists(filename_obj): continue

        # load the raw mesh
        mesh = trimesh.load(filename_raw, force='mesh')
        
        # Split mesh into connected components
        mesh_components = mesh.split(only_watertight=False)

        # sort by vertices number to keep the largest 2
        if len(mesh_components) > 2:
            sorted_components = sorted(mesh_components, key=lambda x: len(x.vertices), reverse=True)
            if 'mask' in filename:
                # Keep the two largest components
                kept_components = sorted_components[:2]
                # Find and keep one cube-like component if it exists
                for component in sorted_components[2:]:
                    if component.vertices.shape == (8, 3) and component.faces.shape == (12, 3):
                        kept_components.append(component)
                mesh_components = kept_components
            else:
                mesh_components = sorted_components[:2]
        # import pdb; pdb.set_trace()
        
        # mesh = trimesh.load("/root/octfusion/data/mask_crown_350/mesh_obj/pair_crown/data0001/model.obj", force='mesh')
        # _ = mesh.export("/root/data0001.obj")
        vertices = mesh.vertices
        bbmin, bbmax = vertices.min(0), vertices.max(0)  # 每次都计算

        if 'mask' in filename:
            center = (bbmin + bbmax) * 0.5
            scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
            # Apply the same transformation to all components
            for component in mesh_components:
                component.vertices = (component.vertices - center) * scale
            center_merge = center
            scale_merge = scale
        else:
            if center_merge is None or scale_merge is None:
                raise RuntimeError("center_merge/scale_merge 未初始化，缺少含 'merg' 的基准样本。")
            # Apply the same transformation to all components
            for component in mesh_components:
                component.vertices = (component.vertices - center_merge) * scale_merge

        # Process each component and combine the results
        sdfs = []
        meshes_new = []
        for component in mesh_components:
            if len(component.faces) == 0: continue
            sdf_comp, mesh_new_comp = mesh2sdf.compute(
                component.vertices, component.faces, size, fix=True, level=level, return_mesh=True)
            sdfs.append(sdf_comp)
            meshes_new.append(mesh_new_comp)

        # Combine SDFs by taking the minimum value (union of shapes)
        sdf = np.min(np.array(sdfs), axis=0)
        
        # Combine meshes
        mesh_new = trimesh.util.concatenate(meshes_new)
        
        mesh_new.vertices = mesh_new.vertices * shape_scale

        # save
        np.savez(filename_box, bbmax=bbmax, bbmin=bbmin, mul=mesh_scale)
        np.save(filename_npy, sdf)
        mesh_new.export(filename_obj)

        # import pdb; pdb.set_trace()


def run_mesh2sdf_split_mp():
    r''' Converts split meshes from crown/oppose to SDFs and manifold meshes with multiprocessing.
    '''

    print('-> Run mesh2sdf split (mp).')
    mesh_scale = 0.8
    filenames = get_filenames('mask_crown_350.txt')
    if not filenames:
        print('No meshes to process.')
        return

    GROUP_SIZE = 3
    if len(filenames) % GROUP_SIZE != 0:
        raise ValueError('mask_crown_350.txt count must be divisible by 3 to keep mask-pair grouping intact.')

    num_meshes = len(filenames) // GROUP_SIZE
    # num_processes = min(32, mp.cpu_count(), num_meshes)
    num_processes = 8
    base_count = num_meshes // num_processes
    remainder = num_meshes % num_processes

    def process(process_id):
        if process_id < remainder:
            idx_start = process_id * (base_count + 1)
            idx_end = idx_start + base_count + 1
        else:
            idx_start = process_id * base_count + remainder
            idx_end = idx_start + base_count

        center_merge = None
        scale_merge = None

        for i in tqdm(range(idx_start * GROUP_SIZE, idx_end * GROUP_SIZE), ncols=80, position=process_id):
            filename = filenames[i]
            filename_raw = os.path.join(
                    file_folder, 'mesh_obj', filename, 'model.obj')
            filename_obj = os.path.join(root_folder, 'mesh', filename + '.obj')
            filename_box = os.path.join(root_folder, 'bbox', filename + '.npz')
            filename_npy = os.path.join(root_folder, 'sdf', filename + '.npy')
            check_folder([filename_obj, filename_box, filename_npy])

            mesh = trimesh.load(filename_raw, force='mesh')
            mesh_components = mesh.split(only_watertight=False)

            if len(mesh_components) > 2:
                sorted_components = sorted(mesh_components, key=lambda x: len(x.vertices), reverse=True)
                if 'mask' in filename:
                    kept_components = sorted_components[:2]
                    for component in sorted_components[2:]:
                        if component.vertices.shape == (8, 3) and component.faces.shape == (12, 3):
                            kept_components.append(component)
                    mesh_components = kept_components
                else:
                    mesh_components = sorted_components[:2]

            vertices = mesh.vertices
            bbmin, bbmax = vertices.min(0), vertices.max(0)

            if 'mask' in filename:
                center = (bbmin + bbmax) * 0.5
                scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
                for component in mesh_components:
                    component.vertices = (component.vertices - center) * scale
                center_merge = center
                scale_merge = scale
            else:
                if center_merge is None or scale_merge is None:
                    raise RuntimeError("center_merge/scale_merge 未初始化，缺少含 'merg' 的基准样本。")
                for component in mesh_components:
                    component.vertices = (component.vertices - center_merge) * scale_merge

            sdfs = []
            meshes_new = []
            for component in mesh_components:
                if len(component.faces) == 0:
                    continue
                sdf_comp, mesh_new_comp = mesh2sdf.compute(
                        component.vertices, component.faces, size, fix=True, level=level, return_mesh=True)
                sdfs.append(sdf_comp)
                meshes_new.append(mesh_new_comp)

            sdf = np.min(np.array(sdfs), axis=0)
            mesh_new = trimesh.util.concatenate(meshes_new)
            mesh_new.vertices = mesh_new.vertices * shape_scale

            np.savez(filename_box, bbmax=bbmax, bbmin=bbmin, mul=mesh_scale)
            np.save(filename_npy, sdf)
            mesh_new.export(filename_obj)

    processes = [mp.Process(target=process, args=[pid]) for pid in range(num_processes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def run_mesh2sdf():
    r''' Converts the meshes from crown/oppose to SDFs and manifold meshes.
    '''

    print('-> Run mesh2sdf.')
    mesh_scale = 0.8
    filenames = get_filenames('closed_crown_350_2.txt')
    # for i in tqdm(range(args.start, args.end), ncols=80):

    center_merge = None
    scale_merge = None

    for i in range(len(filenames)):
        filename = filenames[i]
        # filename_raw = os.path.join(
                # file_folder, 'mesh_Object', filename, 'model.obj')
        filename_raw = os.path.join(
                file_folder,'mesh_obj', filename, 'model.obj')
        filename_obj = os.path.join(root_folder, 'mesh', filename + '.obj')
        filename_box = os.path.join(root_folder, 'bbox', filename + '.npz')
        filename_npy = os.path.join(root_folder, 'sdf', filename + '.npy')
        check_folder([filename_obj, filename_box, filename_npy])
        if os.path.exists(filename_obj): continue

        # load the raw mesh
        mesh = trimesh.load(filename_raw, force='mesh')
        # mesh = trimesh.load("/root/octfusion/data/mask_crown_350/mesh_obj/pair_crown/data0001/model.obj", force='mesh')
        # mesh = trimesh.load("/root/octfusion/data/mask_crown_350/upper_crown/data0003/model.obj", force='mesh')
        # mesh_components = mesh.split(only_watertight=False)
        # import pdb; pdb.set_trace()
        # _ = mesh.export("/root/data0001.obj")
        vertices = mesh.vertices
        bbmin, bbmax = vertices.min(0), vertices.max(0)  # 每次都计算

        if 'mask' in filename:
            center = (bbmin + bbmax) * 0.5
            scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
            vertices = (vertices - center) * scale
            center_merge = center
            scale_merge = scale
        else:
            if center_merge is None or scale_merge is None:
                # Start of modification: Skip if center_merge/scale_merge is not initialized
                print(f"Warning: Skipping {filename} because center_merge/scale_merge is not initialized.")
                continue
                # raise RuntimeError("center_merge/scale_merge 未初始化，缺少含 'merg' 的基准样本。")
                # End of modification
            vertices = (vertices - center_merge) * scale_merge

        # run mesh2sdf
        # Start of modification: Add try-except block to handle mesh2sdf errors
        try:
            sdf, mesh_new = mesh2sdf.compute(vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
        # End of modification

        # import pdb; pdb.set_trace()
        # _ = mesh_new.export("/root/data0001.obj")
        mesh_new.vertices = mesh_new.vertices * shape_scale

        # save
        np.savez(filename_box, bbmax=bbmax, bbmin=bbmin, mul=mesh_scale)
        np.save(filename_npy, sdf)
        mesh_new.export(filename_obj)

def run_mesh2sdf_mp():
    r''' Converts the meshes from Crown/mesh_oppose to SDFs and manifold meshes.
        '''

    print('-> Run mesh2sdf.')
    mesh_scale = 0.8
    filenames = get_filenames('mask_crown_350.txt')

    num_processes = 32
    GROUP_SIZE = 3
    assert len(filenames) % GROUP_SIZE == 0
    num_meshes = len(filenames) // GROUP_SIZE
    base_count = num_meshes // num_processes
    remainder = num_meshes % num_processes

    def process(process_id):
        if process_id < remainder:
            idx_start = process_id * (base_count + 1)
            idx_end = idx_start + base_count + 1
        else:
            idx_start = process_id * base_count + remainder
            idx_end = idx_start + base_count
        
        center_merge = None
        scale_merge = None
        for i in tqdm(range(idx_start * GROUP_SIZE, idx_end * GROUP_SIZE), ncols=80, position=process_id):
            filename = filenames[i]
            filename_raw = os.path.join(
                file_folder,'mesh_obj', filename, 'model.obj')
            filename_obj = os.path.join(root_folder, 'mesh', filename + '.obj')
            filename_box = os.path.join(root_folder, 'bbox', filename + '.npz')
            filename_npy = os.path.join(root_folder, 'sdf', filename + '.npy')
            check_folder([filename_obj, filename_box, filename_npy])
            if os.path.exists(filename_obj): continue

            # load the raw mesh
            mesh = trimesh.load(filename_raw, force='mesh')
            vertices = mesh.vertices
            bbmin, bbmax = vertices.min(0), vertices.max(0)  # 每次都计算
            
            if 'mask' in filename:
                center = (bbmin + bbmax) * 0.5
                scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
                vertices = (vertices - center) * scale
                center_merge = center
                scale_merge = scale
            else:
                if center_merge is None or scale_merge is None:
                    raise RuntimeError("center_merge/scale_merge 未初始化，缺少含 'merg' 的基准样本。")
                vertices = (vertices - center_merge) * scale_merge

            # run mesh2sdf
            sdf, mesh_new = mesh2sdf.compute(vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
            mesh_new.vertices = mesh_new.vertices * shape_scale

            # save
            np.savez(filename_box, bbmax=bbmax, bbmin=bbmin, mul=mesh_scale)
            np.save(filename_npy, sdf)
            mesh_new.export(filename_obj)

    processes = [mp.Process(target=process, args=[pid]) for pid in range(num_processes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def sample_pts_from_mesh_mp():
    r''' Samples 10k points with normals from the ground-truth meshes.
    '''

    print('-> Run sample_pts_from_mesh.')
    num_samples = 100000
    num_interior = 100000
    mesh_folder = os.path.join(root_folder, 'mesh')
    output_folder = os.path.join(root_folder, 'dataset')
    filenames = get_filenames('mask_crown_350_2.txt')

    num_processes = 4
    GROUP_SIZE = 3
    assert len(filenames) % GROUP_SIZE == 0
    num_meshes = len(filenames) // GROUP_SIZE
    base_count = num_meshes // num_processes
    remainder = num_meshes % num_processes

    def process(process_id):
        if process_id < remainder:
            idx_start = process_id * (base_count + 1)
            idx_end = idx_start + base_count + 1
        else:
            idx_start = process_id * base_count + remainder
            idx_end = idx_start + base_count

        for i in tqdm(range(idx_start * GROUP_SIZE, idx_end * GROUP_SIZE), ncols=80, position=process_id):
            filename = filenames[i]
            filename_obj = os.path.join(mesh_folder, filename + '.obj')
            filename_pts = os.path.join(output_folder, filename, 'pointcloud.npz')
            check_folder([filename_pts])
            if os.path.exists(filename_pts): continue

            # sample points
            mesh = trimesh.load(filename_obj, force='mesh')

            # 1. 表面采样
            points, idx = trimesh.sample.sample_surface(mesh, num_samples)
            normals = mesh.face_normals[idx]

            if 'bbox' in filename:
                # 2. 内部采样 (假设是 watertight, 直接用 volume_mesh)
                interior_points = trimesh.sample.volume_mesh(mesh, num_interior)
                # 用最近三角面法向近似内部点的外法向
                _, _, tri_id = mesh.nearest.on_surface(interior_points)
                interior_normals = mesh.face_normals[tri_id]

                # 合并
                points = np.concatenate([points, interior_points], axis=0)
                normals = np.concatenate([normals, interior_normals], axis=0)

            # save points
            np.savez(filename_pts, points=points.astype(np.float16),
                            normals=normals.astype(np.float16))
    
    processes = [mp.Process(target=process, args=[pid]) for pid in range(num_processes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def sample_pts_from_mesh():
    r''' Samples 10k points with normals from the ground-truth meshes.
    '''

    print('-> Run sample_pts_from_mesh.')
    num_samples = 100000
    # num_interior = 100000
    num_interior = 10000
    mesh_folder = os.path.join(root_folder, 'mesh')
    output_folder = os.path.join(root_folder, 'dataset')
    filenames = get_filenames('mask_crown_350_2.txt')
    GROUP_SIZE = 3

    total_groups = (len(filenames) + GROUP_SIZE - 1) // GROUP_SIZE
    start_group = max(0, args.start)
    end_group = total_groups if args.end < 0 else min(args.end, total_groups)

    if start_group >= end_group:
        print('No meshes to process (invalid start/end groups).')
        return

    start_idx = start_group * GROUP_SIZE
    end_idx = min(len(filenames), end_group * GROUP_SIZE)
    print(f'Processing filenames[{start_idx}:{end_idx}) (groups {start_group} to {end_group - 1}).')

    for i in tqdm(range(start_idx, end_idx)):
        filename = filenames[i]
        filename_obj = os.path.join(mesh_folder, filename + '.obj')
        filename_pts = os.path.join(output_folder, filename, 'pointcloud.npz')
        check_folder([filename_pts])
        if os.path.exists(filename_pts):
            print("Skip existing:", filename_pts)
            continue

        # sample points
        mesh = trimesh.load(filename_obj, force='mesh')

        # 1. 表面采样
        points, idx = trimesh.sample.sample_surface(mesh, num_samples)
        normals = mesh.face_normals[idx]

        if 'bbox' in filename:
            # 2. 内部采样 (假设是 watertight, 直接用 volume_mesh)
            interior_points = trimesh.sample.volume_mesh(mesh, num_interior)
            # 用最近三角面法向近似内部点的外法向
            _, _, tri_id = mesh.nearest.on_surface(interior_points)
            interior_normals = mesh.face_normals[tri_id]

            # 合并
            points = np.concatenate([points, interior_points], axis=0)
            normals = np.concatenate([normals, interior_normals], axis=0)

        # save points
        np.savez(filename_pts, points=points.astype(np.float16),
                         normals=normals.astype(np.float16))
        

def sample_sdf():
    r''' Samples ground-truth SDF values for training.
    '''

    # constants
    depth, full_depth = 6, 4
    sample_num = 4    # number of samples in each octree node 也就是文中说的在每个八叉树的节点，采4个点并计算对应的sdf值。
    grid = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    print('-> Sample SDFs from the ground truth.')
    filenames = get_filenames('mask_crown_350_2.txt')
    # for i in tqdm(range(args.start, args.end), ncols=80):
    for i in range(len(filenames)):
        filename = filenames[i]
        dataset_folder = os.path.join(root_folder, 'dataset')
        filename_sdf = os.path.join(root_folder, 'sdf', filename + '.npy')
        filename_pts = os.path.join(dataset_folder, filename, 'pointcloud.npz')
        filename_out = os.path.join(dataset_folder, filename, 'sdf.npz')
        if os.path.exists(filename_out): continue

        # load data
        pts = np.load(filename_pts)
        sdf = np.load(filename_sdf)
        sdf = torch.from_numpy(sdf)
        points = pts['points'].astype(np.float32)
        normals = pts['normals'].astype(np.float32)
        points = points / shape_scale    # rescale points to [-1, 1]

        # build octree
        points = ocnn.octree.Points(torch.from_numpy(points),torch.from_numpy(normals))
        octree = ocnn.octree.Octree(depth = depth, full_depth = full_depth)
        octree.build_octree(points)

        # sample points and grads according to the xyz
        xyzs, grads, sdfs = [], [], []
        for d in range(full_depth, depth + 1):
            xyzb = octree.xyzb(d)
            x,y,z,b = xyzb
            xyz = torch.stack((x,y,z),dim=1).float()

            # sample k points in each octree node
            xyz = xyz.unsqueeze(1) + torch.rand(xyz.shape[0], sample_num, 3)
            xyz = xyz.view(-1, 3)                                    # (N, 3)
            xyz = xyz * (size / 2 ** d)                        # normalize to [0, 2^sdf_depth] 相当于将坐标放大到[0,128]，128是sdf采样的分辨率
            xyz = xyz[(xyz < 127).all(dim=1)]            # remove out-of-bound points
            xyzs.append(xyz)

            # interpolate the sdf values
            xyzi = torch.floor(xyz)                                # the integer part (N, 3)
            corners = xyzi.unsqueeze(1) + grid         # (N, 8, 3)
            coordsf = xyz.unsqueeze(1) - corners     # (N, 8, 3), in [-1.0, 1.0]
            weights = (1 - coordsf.abs()).prod(dim=-1)    # (N, 8, 1)
            corners = corners.long().view(-1, 3)
            x, y, z = corners[:, 0], corners[:, 1], corners[:, 2]
            s = sdf[x, y, z].view(-1, 8)
            sw = torch.sum(s * weights, dim=1)
            sdfs.append(sw)

            # calc the gradient
            gx = s[:, 4] - s[:, 0] + s[:, 5] - s[:, 1] + \
                     s[:, 6] - s[:, 2] + s[:, 7] - s[:, 3]    # noqa
            gy = s[:, 2] - s[:, 0] + s[:, 3] - s[:, 1] + \
                     s[:, 6] - s[:, 4] + s[:, 7] - s[:, 5]    # noqa
            gz = s[:, 1] - s[:, 0] + s[:, 3] - s[:, 2] + \
                     s[:, 5] - s[:, 4] + s[:, 7] - s[:, 6]    # noqa
            grad = torch.stack([gx, gy, gz], dim=-1)
            norm = torch.sqrt(torch.sum(grad ** 2, dim=-1, keepdims=True))
            grad = grad / (norm + 1.0e-8)
            grads.append(grad)

        # concat the results
        xyzs = torch.cat(xyzs, dim=0).numpy()
        points = (xyzs / 64 - 1).astype(np.float16) * shape_scale    # 这里的points是sdf采样点的points，然后继续缩放到[-0.5, 0.5], 真的搞不懂为什么非要加这个0.5的shape_scale转来转去的，有啥意义。
        grads = torch.cat(grads, dim=0).numpy().astype(np.float16)
        sdfs = torch.cat(sdfs, dim=0).numpy().astype(np.float16)     # 这里的sdf还是跟之前一样，都是在[-1, 1]之间

        # save results
        # points = (points * args.scale).astype(np.float16)    # in [-scale, scale]
        np.savez(filename_out, points=points, grad=grads, sdf=sdfs)    # 也就是说sdf的值是在[-1,1]的尺度上，但是point的坐标在[-0.5, 0.5]


def sample_occu():
    r''' Samples occupancy values for evaluating the IoU following ConvONet.
    '''

    num_samples = 100000
    grid = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                     [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    #filenames = get_filenames('test.txt') + get_filenames('test_unseen5.txt')
    filenames = get_filenames('mask_crown_350_2.txt')
    for filename in tqdm(filenames, ncols=80):
        filename_sdf = os.path.join(root_folder, 'sdf', filename + '.npy')
        filename_occu = os.path.join(root_folder, 'dataset', filename, 'points')
        if os.path.exists(filename_occu) or (not os.path.exists(filename_sdf)):
            continue

        sdf = np.load(filename_sdf)
        factor = 127.0 / 128.0    # make sure the interpolation is well defined
        points_uniform = np.random.rand(num_samples, 3) * factor    # in [0, 1)
        points = (points_uniform - 0.5) * (2 * shape_scale)             # !!! rescale
        points = points.astype(np.float16)

        # interpolate the sdf values
        xyz = points_uniform * 128                                             # in [0, 127)
        xyzi = np.floor(xyz)                                                         # the integer part (N, 3)
        corners = np.expand_dims(xyzi, 1) + grid                 # (N, 8, 3)
        coordsf = np.expand_dims(xyz, 1) - corners             # (N, 8, 3), in [-1.0, 1.0]
        weights = np.prod(1 - np.abs(coordsf), axis=-1)    # (N, 8)

        corners = np.reshape(corners.astype(np.int64), (-1, 3))
        x, y, z = corners[:, 0], corners[:, 1], corners[:, 2]
        values = np.reshape(sdf[x, y, z], (-1, 8))
        value = np.sum(values * weights, axis=1)
        occu = value < 0
        occu = np.packbits(occu)

        # save
        np.savez(filename_occu, points=points, occupancies=occu)


def generate_test_points():
    r''' Generates points in `ply` format for testing.
    '''

    noise_std = 0.005
    point_sample_num = 3000
    # filenames = get_filenames('test.txt') + get_filenames('test_unseen5.txt')
    filenames = get_filenames('mask_crown_350_2.txt')
    for filename in tqdm(filenames, ncols=80):
        filename_pts = os.path.join(
                root_folder, 'dataset', filename, 'pointcloud.npz')
        filename_ply = os.path.join(
                root_folder, 'test.input', filename + '.ply')
        if not os.path.exists(filename_pts): continue
        check_folder([filename_ply])

        # sample points
        pts = np.load(filename_pts)
        points = pts['points'].astype(np.float32)
        noise = noise_std * np.random.randn(point_sample_num, 3)
        rand_idx = np.random.choice(points.shape[0], size=point_sample_num)
        points_noise = points[rand_idx] + noise

        # save ply
        vertices = []
        py_types = (float, float, float)
        npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        for idx in range(points_noise.shape[0]):
            vertices.append(
                    tuple(dtype(d) for dtype, d in zip(py_types, points_noise[idx])))
        structured_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(structured_array, 'vertex')
        PlyData([el]).write(filename_ply)

def generate_dataset_unseen5():
    r'''Creates the unseen5 dataset
    '''

    dataset_folder = os.path.join(root_folder, 'dataset')
    unseen5_folder = os.path.join(root_folder, 'dataset.unseen5')
    if not os.path.exists(unseen5_folder):
        os.makedirs(unseen5_folder)
    for folder in ['02808440', '02773838', '02818832', '02876657', '03938244']:
        curr_folder = os.path.join(dataset_folder, folder)
        if os.path.exists(curr_folder):
            shutil.move(os.path.join(dataset_folder, folder), unseen5_folder)


def copy_convonet_filelists():
    r''' Copies the filelist of ConvONet to the datasets, which are needed when
     calculating the evaluation metrics.
     '''

    with open(os.path.join(root_folder, 'filelist/list_crown.txt'), 'r') as fid:
        lines = fid.readlines()
    filenames = [line.split()[0] for line in lines]
    filelist_folder = os.path.join(root_folder, 'filelist')
    for filename in filenames:
        src_name = os.path.join(filelist_folder, filename)
        # des_name = src_name.replace('filelist/convonet.filelist', 'dataset')    \
                                            #  .replace('filelist/unseen5.filelist', 'dataset.unseen5')
        des_name = src_name.replace('filelist/convonet.filelist', 'dataset')
        if not os.path.exists(des_name):
            shutil.copy(src_name, des_name)

def generate_dataset():
    sample_pts_from_mesh()
    # sample_pts_from_mesh_mp()
    sample_sdf()
    sample_occu()
    generate_test_points()
    # generate_dataset_unseen5()
    # copy_convonet_filelists()

if __name__ == '__main__':
    eval('%s()' % args.run)