# Reference: diffusion is borrowed from the LDM repo: https://github.com/CompVis/latent-diffusion
# Specifically, functions from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py

import os
import sys
from collections import OrderedDict
from functools import partial
import copy
import time
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm
from random import random
import ocnn
from ocnn.nn import octree2voxel, octree_pad
from ocnn.octree import Octree, Points
from models.networks.dualoctree_networks import dual_octree

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.special import expm1

from models.base_model import BaseModel
from models.networks.diffusion_networks.graph_unet_union import UNet3DModel
from models.model_utils import load_dualoctree
from models.networks.diffusion_networks.ldm_diffusion_util import *


# distributed
from utils.distributed import reduce_loss_dict, get_rank, get_world_size

# rendering
from utils.util_dualoctree import calc_sdf, octree2split_small, octree2split_large, split2octree_small, split2octree_large
from utils.util import TorchRecoder, seed_everything, category_5_to_label

from utils.points_subset_to_split_mask import points_to_split_mask, points_to_node_mask

TRUNCATED_TIME = 0.7


class OctFusionModel(BaseModel):
    def name(self):
        return 'SDFusion-Model-Union-Two-Times'

    def initialize(self, opt):
        self.network_initialize(opt)
        self.optimizer_initialize(opt)
        
    def network_initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.mode == "train"
        self.model_name = self.name()
        self.device = opt.device
        self.gradient_clip_val = 1.
        self.start_iter = opt.start_iter

        if self.isTrain:
            self.log_dir = os.path.join(opt.logs_dir, opt.name)
            self.train_dir = os.path.join(self.log_dir, 'train_temp')
            self.test_dir = os.path.join(self.log_dir, 'test_temp')


        ######## START: Define Networks ########
        assert opt.df_cfg is not None
        assert opt.vq_cfg is not None

        # init df
        df_conf = OmegaConf.load(opt.df_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)
        self.batch_size = vq_conf.data.train.batch_size = opt.batch_size
        
        self.vq_conf = vq_conf
        self.solver = self.vq_conf.solver

        self.input_depth = self.vq_conf.model.depth
        self.octree_depth = self.vq_conf.model.depth_stop
        self.small_depth = 6
        self.large_depth = 8
        self.full_depth = self.vq_conf.model.full_depth

        self.load_octree = self.vq_conf.data.train.load_octree
        self.load_pointcloud = self.vq_conf.data.train.load_pointcloud
        self.load_split_small = self.vq_conf.data.train.load_split_small

        # init diffusion networks
        df_model_params = df_conf.model.params
        unet_params = df_conf.unet.params
        self.conditioning_key = df_model_params.conditioning_key
        self.num_timesteps = df_model_params.timesteps
        self.enable_label = "num_classes" in df_conf.unet.params
        self.df_type = unet_params.df_type

        self.df = UNet3DModel(opt.stage_flag, **unet_params)
        self.df.to(self.device)
        self.stage_flag = opt.stage_flag

        # record z_shape
        self.split_channel = 8
        self.code_channel = self.vq_conf.model.embed_dim
        z_sp_dim = 2 ** self.full_depth
        self.z_shape = (self.split_channel, z_sp_dim, z_sp_dim, z_sp_dim)

        self.ema_df = copy.deepcopy(self.df)
        self.ema_df.to(self.device)
        if opt.isTrain:
            self.ema_rate = opt.ema_rate
            self.ema_updater = EMA(self.ema_rate)
            self.reset_parameters()
            set_requires_grad(self.ema_df, False)

        self.noise_schedule = "linear"
        if self.noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif self.noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {self.noise_schedule}')

        # init vqvae

        self.autoencoder = load_dualoctree(conf = vq_conf, ckpt = opt.vq_ckpt, opt = opt)

        ######## END: Define Networks ########

    def optimizer_initialize(self, opt):
        if opt.pretrain_ckpt is not None:
            self.load_ckpt(opt.pretrain_ckpt, self.df, self.ema_df, load_options=["unet_lr"])
        
        if self.stage_flag == "lr":
            self.set_requires_grad([
                self.df.unet_hr
            ], False)
        elif self.stage_flag == "hr":
            self.set_requires_grad([
                self.df.unet_lr
            ], False)
        
        if self.isTrain:

            # initialize optimizers
            self.optimizer = optim.AdamW([p for p in self.df.parameters() if p.requires_grad == True], lr=opt.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        # if opt.ckpt is None and os.path.exists(os.path.join(opt.logs_dir, opt.name, "ckpt/df_steps-latest.pth")):
        #     opt.ckpt = os.path.join(opt.logs_dir, opt.name, "ckpt/df_steps-latest.pth")
        
        if opt.ckpt is not None:
            if self.stage_flag == "lr":
                load_options = ["unet_lr"]
            elif self.stage_flag == "hr":
                load_options = ["unet_lr", "unet_hr"]
            if self.isTrain:
                load_options.append("opt")
            self.load_ckpt(opt.ckpt, self.df, self.ema_df, load_options)
                
        trainable_params_num = 0
        for m in [self.df]:
            trainable_params_num += sum([p.numel() for p in m.parameters() if p.requires_grad == True])
        print("Trainable_params: ", trainable_params_num)

        # for distributed training
        if self.opt.distributed:
            self.make_distributed(opt)
            self.df_module = self.df.module
            self.autoencoder_module = self.autoencoder.module

        else:
            self.df_module = self.df
            self.autoencoder_module = self.autoencoder

    def reset_parameters(self):
        self.ema_df.load_state_dict(self.df.state_dict())

    def make_distributed(self, opt):
        self.df = nn.parallel.DistributedDataParallel(
            self.df,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        if opt.sync_bn:
            self.autoencoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.autoencoder)
        self.autoencoder = nn.parallel.DistributedDataParallel(
            self.autoencoder,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    ############################ START: init diffusion params ############################

    def batch_to_cuda(self, batch):
        # p batch['points'][0].points.shape
        # torch.Size([100000, 3])
        # p batch['normals'][0].normals.shape
        # torch.Size([100000, 3])
        def points2octree(points):
            octree = ocnn.octree.Octree(depth = self.input_depth, full_depth = self.full_depth)
            octree.build_octree(points)
            return octree

        if self.load_pointcloud:
            points = [pts.cuda(non_blocking=True) for pts in batch['points']]
            octrees = [points2octree(pts) for pts in points]
            octree = ocnn.octree.merge_octrees(octrees)
            octree.construct_all_neigh()
            batch['octree_in'] = octree
            batch['split_small'] = octree2split_small(batch['octree_in'], self.full_depth)
            # p batch['split_small'].shape
            # torch.Size([2, 8, 16, 16, 16]) # scale to [-1, 1]

        batch['label'] = batch['label'].cuda()
        if self.load_octree:
            batch['octree_in'] = batch['octree_in'].cuda()
            batch['split_small'] = octree2split_small(batch['octree_in'], self.full_depth)
            # batch['split_large'] = self.octree2split_large(batch['octree_in'])
        elif self.load_split_small:
            batch['split_small'] = batch['split_small'].cuda()
            batch['octree_in'] = split2octree_small(batch['split_small'], self.input_depth, self.full_depth)

    def set_input(self, input=None):
        self.batch_to_cuda(input)
        self.split_small = input['split_small']
        # self.split_large = input['split_large']
        self.octree_in = input['octree_in']
        self.batch_size = self.octree_in.batch_size

        # simulate
        # input['points_subset'] = torch.rand(self.batch_size, 1000, 3).to(input['split_small']) * 2 - 1  # [-1, 1]
        
        # input['filename']: pair_crown/0001/data0001 这个是旧的
        # input['filename']: pair_crown/data0001 这个是新的
        suffixs = []
        for filename in input['filename']:
            suffix = filename.split('/')[-1]
            suffixs.append(suffix)
        bbox_names = []
        oppose_crown_names = []
        for suffix in suffixs:
            bbox_name = '/root/octfusion/data/' + self.opt.category + '/dataset/' + 'bbox_crown/' + suffix + '/pointcloud.npz'
            # oppose_crown_name = '/root/octfusion/data/' + self.opt.category + '/dataset/' + 'oppose_crown/' + suffix + '/pointcloud.npz'
            bbox_names.append(bbox_name)
            # oppose_crown_names.append(oppose_crown_name)

        # NOTE: we load bounding box here
        # raw = np.load("/root/octfusion/data/Crown/dataset/gbox/boundingbox/pointcloud.npz")
        # points = raw['points']
        # point_scale = 0.5
        # points = points / point_scale    # scale to [-1.0, 1.0]
        # points = torch.from_numpy(points).float().to(input['split_small'])
        # points.clip(min=-1, max=1)
        # input['points_subset'] = points.unsqueeze(0).repeat(self.batch_size, 1, 1)  # [-1, 1]
        bboxes = []
        # NUM_POINTS = 200000
        NUM_POINTS = 110000
        for bbox_name in bbox_names:
            raw = np.load(bbox_name)
            points = raw['points']
            point_scale = 0.5
            points = points / point_scale    # scale to [-1.0, 1.0]
            points = torch.from_numpy(points).float().to(input['split_small'])
            points.clip(min=-1, max=1)
            if points.shape[0] < NUM_POINTS:
                last_point = points[-1].unsqueeze(0)  # Shape: [1, 3]
                repeated_points = last_point.repeat(NUM_POINTS - points.shape[0], 1)
                points = torch.cat([points, repeated_points], dim=0)
            assert points.shape[0] == NUM_POINTS, f"points.shape[0] = {points.shape[0]}, expected {NUM_POINTS}"
            bboxes.append(points)
        input['points_subset'] = torch.stack(bboxes, dim=0)  # [-1, 1]
        self.points_subset = input['points_subset']

        # cloud = trimesh.points.PointCloud(raw['points'])
        # cloud.export("/root/octfusion/logs/mytest_union/union_2t_test_lr2e-4/results_recover_obj/bbox.obj")
        # import pdb; pdb.set_trace()

        # NOTE: we load oppose crown here
        # 现在不用到oppose
        # oppose_crowns = []
        # for oppose_crown_name in oppose_crown_names:
        #     raw = np.load(oppose_crown_name)
        #     points = raw['points']
        #     point_scale = 0.5
        #     points = points / point_scale    # scale to [-1.0, 1.0]
        #     points = torch.from_numpy(points).float().to(input['split_small'])
        #     points.clip(min=-1, max=1)
        #     oppose_crowns.append(points)
        # input['oppose_crown'] = torch.stack(oppose_crowns, dim=0)

        from utils.points_subset_to_split_mask import points_to_split_mask
        # input['points_subset'] 形状: [batch_size, N, 3]
        self.subset_mask = points_to_split_mask(
            input['points_subset'], 
            self.octree_in, 
            self.full_depth
        )
        # oppose_mask = points_to_split_mask(
        #     input['oppose_crown'], 
        #     self.octree_in, 
        #     self.full_depth
        # )
        # self.subset_mask = self.subset_mask * (1 - oppose_mask)  # 排除掉 oppose crown 部分

        # print(f"Generated subset mask: {self.subset_mask.shape}")

        if self.enable_label:
            self.label = input['label']
        else:
            self.label = None

        # (1 - self.subset_mask)为0的位置，将self.split_small对应位置设为-1
        mask = (1 - self.subset_mask)
        split_small_masked = self.split_small.clone()
        split_small_masked[mask == 0] = -1

        return self.split_small, 2 * self.subset_mask - 1, split_small_masked

    def switch_train(self):
        self.df.train()

    def switch_eval(self):
        self.df.eval()


    def calc_loss(self, input_data, doctree_in, batch_id, unet_type, unet_lr, df_type="x0", mask=None):
        times = torch.zeros(
            (self.batch_size,), device=self.device).float().uniform_(0, 1)
        
        noise = torch.randn_like(input_data)

        noise_level = self.log_snr(times)
        alpha, sigma = log_snr_to_alpha_sigma(noise_level)
        batch_alpha = right_pad_dims_to(input_data, alpha[batch_id])
        batch_sigma = right_pad_dims_to(input_data, sigma[batch_id])
        noised_data = batch_alpha * input_data + batch_sigma * noise

        output = self.df(unet_type=unet_type, x=noised_data, doctree=doctree_in, unet_lr=unet_lr, timesteps=noise_level, label=self.label)
        
        if df_type == "x0":
            loss = F.mse_loss(output, input_data, reduction='none')
        elif df_type == "eps":
            # x_start = (noised_data - output * batch_sigma) / batch_alpha.clamp(min=1e-8)
            # self.output = self.autoencoder_module.decode_code(x_start, doctree_in)
            # self.get_sdfs(self.output['neural_mpu'], self.batch_size, bbox = None)
            # self.export_mesh(save_dir = "mytools/octree", index = 0)

            # self.output = self.autoencoder_module.decode_code(input_data, doctree_in)
            # self.get_sdfs(self.output['neural_mpu'], self.batch_size, bbox = None)
            # self.export_mesh(save_dir = "mytools/octree", index = 2)
            loss = F.mse_loss(output, noise, reduction='none')
        else:
            raise ValueError(f'invalid loss type {df_type}')
        
        # 应用 mask：只对 mask=1 的区域计算损失
        # import pdb; pdb.set_trace()
        if mask is not None:
            loss = loss * mask
            # 计算有效区域的平均损失
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
            
        return loss

    def forward(self):

        self.df.train()

        c = None        

        self.df_hr_loss = torch.tensor(0., device=self.device)
        self.df_lr_loss = torch.tensor(0., device=self.device)

        if self.stage_flag == "lr":
            # self.df_lr_loss = self.forward_lr(split_small)
            batch_id = torch.arange(0, self.batch_size, device=self.device).long()
            self.df_lr_loss = self.calc_loss(self.split_small, None, batch_id, "lr", None, self.df_type[0], mask=self.subset_mask)
            
        elif self.stage_flag == "hr":
            with torch.no_grad():
                self.input_data, self.doctree_in = self.autoencoder_module.extract_code(self.octree_in)
            # self.df_hr_loss = self.forward_hr(self.input_data, self.small_depth, "hr", self.df_module.unet_lr)
            hr_mask = points_to_node_mask(
                self.points_subset, 
                self.octree_in, 
                self.small_depth,
                self.full_depth
            )
            # Broadcast mask to match input_data channels for correct loss normalization
            self.hr_mask = hr_mask.repeat(1, self.input_data.shape[1])
            
            # NOTE: we do not use mask for hr stage
            hr_mask = None
            self.hr_mask = None

            self.df_hr_loss = self.calc_loss(self.input_data, self.doctree_in, self.doctree_in.batch_id(self.small_depth), "hr", self.df_module.unet_lr, self.df_type[1], mask=self.hr_mask)

        self.loss = self.df_lr_loss + self.df_hr_loss

    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    def sample_loop(self, doctree_lr = None, ema=False, shape=None, ddim_steps=200, label=None, unet_type="lr", unet_lr=None, df_type="x0", truncated_index=0.0, mask=None, original_data=None):
        batch_size = self.vq_conf.data.test.batch_size

        time_pairs = self.get_sampling_timesteps(
            batch_size, device=self.device, steps=ddim_steps)

        noised_data = torch.randn(shape, device = self.device)

        # 处理 mask / original_data 的前置检查与准备
        use_inpaint = (mask is not None) and (original_data is not None)
        if use_inpaint:
            # 强制 mask 为 float tensor，且与 noised_data 同 device/shape 可广播
            mask = mask.to(noised_data.dtype).to(self.device)
            # 预采样一个固定的 eps_orig（对整个采样过程保持一致）
            eps_orig = torch.randn_like(original_data)
            # 初始化时用第一个 time 的 alpha/sigma 给 original_data 加噪 (consistent init)
            # 取第一个 time 对应的 t（time_pairs 是一系列 (t,t_next)；第一个元素的 t 即 time_pairs[0][0]）
            t0, _ = time_pairs[0]
            padded_log_snr_t0 = right_pad_dims_to(noised_data, self.log_snr(t0))
            alpha_t0, sigma_t0 = log_snr_to_alpha_sigma(padded_log_snr_t0)
            # original_noised_t0 广播成和 noised_data 相同形状
            original_noised_t0 = alpha_t0 * original_data + sigma_t0 * eps_orig
            # masked 区域使用随机噪声（noised_data 已是 randn）；unmasked 用 original_noised_t0
            noised_data = mask * noised_data + (1.0 - mask) * original_noised_t0
        else:
            eps_orig = None  # for clarity

        x_start = None

        time_iter = tqdm(time_pairs, desc='small sampling loop time step')

        for t, t_next in time_iter:

            log_snr = self.log_snr(t)
            log_snr_next = self.log_snr(t_next)
            noise_cond = log_snr

            if ema:
                output = self.ema_df(unet_type=unet_type, x=noised_data, doctree=doctree_lr,  timesteps=noise_cond, unet_lr=unet_lr, x_self_cond=x_start, label=label)
            else:
                output = self.df(unet_type=unet_type, x=noised_data, doctree=doctree_lr,  timesteps=noise_cond, unet_lr=unet_lr, x_self_cond=x_start, label=label)

            if t[0] < truncated_index and unet_type == "lr":
                output.sign_()

            if df_type == "x0":
                x_start = output
                padded_log_snr, padded_log_snr_next = map(
                    partial(right_pad_dims_to, noised_data), (log_snr, log_snr_next))

                alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
                alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

                c = -expm1(padded_log_snr - padded_log_snr_next)
                mean = alpha_next * (noised_data * (1 - c) / alpha + c * output)
                variance = (sigma_next ** 2) * c
                noise = torch.where(
                    # rearrange(t_next > truncated_index, 'b -> b 1 1 1 1'),
                    right_pad_dims_to(noised_data, t_next > truncated_index),
                    torch.randn_like(noised_data),
                    torch.zeros_like(noised_data)
                )
                noised_data_new = mean + torch.sqrt(variance) * noise
                
                # 如果做 inpainting：用相同 eps_orig 构造 original_noised（而不是每步随机）
                if use_inpaint:
                    # 用 eps_orig 和 alpha_next/sigma_next 构造 consistent 的 original_noised
                    original_noised = alpha_next * original_data + sigma_next * eps_orig
                    noised_data = mask * noised_data_new + (1.0 - mask) * original_noised
                else:
                    noised_data = noised_data_new

            elif df_type == "eps":
                alpha, sigma = log_snr_to_alpha_sigma(log_snr)
                alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)
                alpha, sigma, alpha_next, sigma_next = alpha[0], sigma[0], alpha_next[0], sigma_next[0]
                x_start = (noised_data - output * sigma) / alpha.clamp(min=1e-8)
                noised_data_new = x_start * alpha_next + output * sigma_next
                
                if use_inpaint:
                    original_noised = alpha_next * original_data + sigma_next * eps_orig
                    noised_data = mask * noised_data_new + (1.0 - mask) * original_noised
                else:
                    noised_data = noised_data_new
        
        return noised_data
    
    @torch.no_grad()
    def sample(self, split_small = None, category = 'airplane', prefix = 'results', ema = False, ddim_steps=200, clean = False, save_index = 0, mask=None, original_data=None):

        if ema:
            self.ema_df.eval()
        else:
            self.df.eval()
        
        batch_size = self.vq_conf.data.test.batch_size
        if self.enable_label:
            label = torch.ones(batch_size).to(self.device) * category_5_to_label[category]
            label = label.long()
        else:
            label = None
            
        save_dir = os.path.join(self.opt.logs_dir, self.opt.name, f"{prefix}_{category}")
        
        if split_small == None:
            seed_everything(self.opt.seed + save_index)
            # 对于 lr 阶段，传递相应的 mask 和 original_data
            lr_mask = mask if mask is not None else self.subset_mask
            lr_original = original_data if original_data is not None else self.split_small
            print(f"Sampling LR stage with mask: {'provided mask' if mask is not None else 'subset mask from input'}")
            print(f"Sampling LR stage with original_data: {'provided original_data' if original_data is not None else 'input split_small'}")
            split_small = self.sample_loop(doctree_lr=None, ema=ema, shape=(batch_size, *self.z_shape), ddim_steps=ddim_steps, label=label, unet_type="lr", unet_lr=None, df_type=self.df_type[0], truncated_index=TRUNCATED_TIME, mask=lr_mask, original_data=lr_original)
        
        octree_small = split2octree_small(split_small, self.octree_depth, self.full_depth)
        self.export_octree(octree_small, depth = self.small_depth, save_dir = os.path.join(save_dir, "octree"), index = save_index)
        # for i in range(batch_size):
        #     save_path = os.path.join(save_dir, "splits_small", f"{save_index}.pth")
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     torch.save(split_small[i].unsqueeze(0), save_path)
        
        if self.stage_flag == "lr":
            return
        
        doctree_small = dual_octree.DualOctree(octree_small)
        doctree_small.post_processing_for_docnn()

        doctree_small_num = doctree_small.total_num
        
        # Prepare HR mask and original data
        hr_mask = None
        hr_original_data = None
        
        # Calculate HR original data
        with torch.no_grad():
            self.input_data, self.doctree_in = self.autoencoder_module.extract_code(self.octree_in)
        # hr_mask = points_to_node_mask(
        #     self.points_subset, 
        #     self.octree_in, 
        #     self.small_depth,
        #     self.full_depth
        # )
        hr_mask = points_to_node_mask(
            self.points_subset, 
            doctree_small.octree, 
            self.small_depth,
            self.full_depth
        )
        # Broadcast mask to match input_data channels for correct loss normalization
        self.hr_mask = hr_mask.repeat(1, self.input_data.shape[1])

        # NOTE: no use of mask for hr
        hr_mask = None
        self.hr_mask = None
        
        hr_original_data = self.get_hr_original_data(doctree_small, self.octree_in, self.input_data)
        # hr_original_data = self.input_data
        
        seed_everything(self.opt.seed)
        samples = self.sample_loop(doctree_lr=doctree_small, shape=(doctree_small_num, self.code_channel), ema=ema, ddim_steps=ddim_steps, label=label, unet_type="hr", unet_lr=self.ema_df.unet_lr, df_type=self.df_type[1], mask=hr_mask, original_data=hr_original_data)

        print(samples.max())
        print(samples.min())
        print(samples.mean())
        print(samples.std())

        # decode z
        self.output = self.autoencoder_module.decode_code(samples, doctree_small)
        self.get_sdfs(self.output['neural_mpu'], batch_size, bbox = None)
        self.export_mesh(save_dir = save_dir, index = save_index, clean = clean)

    def export_octree(self, octree, depth, save_dir = None, index = 0):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileExistsError:
            pass

        batch_id = octree.batch_id(depth = depth, nempty = False)
        data = torch.ones((len(batch_id), 1), device = self.device)
        data = octree2voxel(data = data, octree = octree, depth = depth, nempty = False)
        data = data.permute(0,4,1,2,3).contiguous()

        batch_size = octree.batch_size

        for i in tqdm(range(batch_size)):
            voxel = data[i].squeeze().cpu().numpy()
            mesh = voxel2mesh(voxel)
            if batch_size == 1:
                mesh.export(os.path.join(save_dir, f'{index}.obj'))
            else:
                mesh.export(os.path.join(save_dir, f'{index + i}.obj'))


    def get_sdfs(self, neural_mpu, batch_size, bbox):
        # bbox used for marching cubes
        if bbox is not None:
            self.bbmin, self.bbmax = bbox[:3], bbox[3:]
        else:
            sdf_scale = self.solver.sdf_scale
            self.bbmin, self.bbmax = -sdf_scale, sdf_scale    # sdf_scale = 0.9

        self.sdfs = calc_sdf(neural_mpu, batch_size, size = self.solver.resolution, bbmin = self.bbmin, bbmax = self.bbmax)

    def export_mesh(self, save_dir, index = 0, level = 0, clean = False):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileExistsError:
            pass
        ngen = self.sdfs.shape[0]
        size = self.solver.resolution
        mesh_scale=self.vq_conf.data.test.point_scale
        for i in range(ngen):
            filename = os.path.join(save_dir, f'{index + i}.obj')
            if ngen == 1:
                filename = os.path.join(save_dir, f'{index}.obj')
            sdf_value = self.sdfs[i].cpu().numpy()
            vtx, faces = np.zeros((0, 3)), np.zeros((0, 3))
            try:
                vtx, faces, _, _ = skimage.measure.marching_cubes(sdf_value, level)
            except:
                pass
            if vtx.size == 0 or faces.size == 0:
                print('Warning from marching cubes: Empty mesh!')
                return
            vtx = vtx * ((self.bbmax - self.bbmin) / size) + self.bbmin   # [0,sz]->[bbmin,bbmax]  把vertex放缩到[bbmin, bbmax]之间
            vtx = vtx * mesh_scale
            mesh = trimesh.Trimesh(vtx, faces)  # 利用Trimesh创建mesh并存储为obj文件。
            if clean:
                components = mesh.split(only_watertight=False)
                bbox = []
                for c in components:
                    bbmin = c.vertices.min(0)
                    bbmax = c.vertices.max(0)
                    bbox.append((bbmax - bbmin).max())
                max_component = np.argmax(bbox)
                mesh = components[max_component]
            mesh.export(filename)


    def backward(self):

        self.loss.backward()

    def update_EMA(self):
        update_moving_average(self.ema_df, self.df, self.ema_updater)

    def optimize_parameters(self):

        # self.set_requires_grad([self.df.unet_hr], requires_grad=True)

        self.forward()
        assert not torch.isnan(self.loss).any()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        self.update_EMA()

    def get_current_errors(self):

        ret = OrderedDict([
            ('loss', self.loss.data),
            ('lr', self.optimizer.param_groups[0]['lr']),
        ])

        if hasattr(self, 'loss_gamma'):
            ret['gamma'] = self.loss_gamma.data

        return ret

    def save(self, label, global_iter):

        state_dict = {
            'df_unet_lr': self.df_module.unet_lr.state_dict(),
            'ema_df_unet_lr': self.ema_df.unet_lr.state_dict(),
            'opt': self.optimizer.state_dict(),
            'global_step': global_iter,
        }
        if self.stage_flag == "hr":
            state_dict['df_unet_hr'] = self.df_module.unet_hr.state_dict()
            state_dict['ema_df_unet_hr'] = self.ema_df.unet_hr.state_dict()

        save_filename = 'df_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        ckpts = os.listdir(self.opt.ckpt_dir)
        ckpts = [ck for ck in ckpts if ck!='df_steps-latest.pth']
        ckpts.sort(key=lambda x: int(x[9:-4]))
        if len(ckpts) > self.opt.ckpt_num:
            for ckpt in ckpts[:-self.opt.ckpt_num]:
                os.remove(os.path.join(self.opt.ckpt_dir, ckpt))

        torch.save(state_dict, save_path)

    def load_ckpt(self, ckpt, df, ema_df, load_options=[]):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt
        
        if "unet_lr" in load_options and "df_unet_lr" in state_dict:
            df.unet_lr.load_state_dict(state_dict['df_unet_lr'])
            ema_df.unet_lr.load_state_dict(state_dict['ema_df_unet_lr'])
            print(colored('[*] weight successfully load unet_lr from: %s' % ckpt, 'blue'))
        if "unet_hr" in load_options and "df_unet_hr" in state_dict:
            df.unet_hr.load_state_dict(state_dict['df_unet_hr'])
            ema_df.unet_hr.load_state_dict(state_dict['ema_df_unet_hr'])
            print(colored('[*] weight successfully load unet_hr from: %s' % ckpt, 'blue'))

        if "opt" in load_options and "opt" in state_dict:
            self.start_iter = state_dict['global_step']
            print(colored('[*] training start from: %d' % self.start_iter, 'green'))
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))

    def get_hr_original_data(self, doctree_small, octree_in, input_data):
        device = input_data.device
        code_dim = input_data.shape[1]
        
        # 1. Collect keys for octree_in (GT)
        gt_keys = []
        for d in range(self.full_depth, self.small_depth):
            children = octree_in.children[d]
            is_leaf = children < 0
            keys = octree_in.keys[d]
            leaf_keys = keys[is_leaf]
            full_keys = leaf_keys | (torch.tensor(d, device=device) << 58)
            gt_keys.append(full_keys)
            
        keys = octree_in.keys[self.small_depth]
        full_keys = keys | (torch.tensor(self.small_depth, device=device) << 58)
        gt_keys.append(full_keys)
        
        all_gt_keys = torch.cat(gt_keys)
        
        # 2. Collect keys for doctree_small (Target)
        target_keys = []
        octree_small = doctree_small.octree
        for d in range(self.full_depth, self.small_depth):
            children = octree_small.children[d]
            is_leaf = children < 0
            keys = octree_small.keys[d]
            leaf_keys = keys[is_leaf]
            full_keys = leaf_keys | (torch.tensor(d, device=device) << 58)
            target_keys.append(full_keys)
            
        keys = octree_small.keys[self.small_depth]
        full_keys = keys | (torch.tensor(self.small_depth, device=device) << 58)
        target_keys.append(full_keys)
        
        all_target_keys = torch.cat(target_keys)
        
        # 3. Match
        sorted_gt_keys, gt_indices = torch.sort(all_gt_keys)
        positions = torch.searchsorted(sorted_gt_keys, all_target_keys)
        positions = torch.clamp(positions, 0, len(sorted_gt_keys) - 1)
        
        matched_keys = sorted_gt_keys[positions]
        is_match = (matched_keys == all_target_keys)
        matched_indices = gt_indices[positions]
        
        result = torch.zeros(len(all_target_keys), code_dim, device=device)
        if is_match.any():
            result[is_match] = input_data[matched_indices[is_match]]
            
        return result
