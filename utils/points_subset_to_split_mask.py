# --------------------------------------------------------
# 简单实现：从点云子集生成split_small格式的mask  
# 使用octree.search_xyzb()函数直接查找坐标对应位置
# --------------------------------------------------------

import torch

def points_to_split_mask(points_subset, octree, full_depth):
    """
    从点云子集生成与split_small相同形状的mask
    
    Args:
        points_subset (torch.Tensor): 子集点坐标 [batch_size, N, 3], 范围 [-1, 1]  
        octree (Octree): octree结构
        full_depth (int): full_depth参数
        
    Returns:
        mask (torch.Tensor): [batch_size, 8, 2^full_depth, 2^full_depth, 2^full_depth]
    """
    batch_size, N, _ = points_subset.shape
    grid_size = 2 ** full_depth
    
    if N == 0:
        return torch.zeros(batch_size, 8, grid_size, grid_size, grid_size, 
                          dtype=torch.float32, device=points_subset.device)
    
    # 1. 转换坐标到octree范围 [0, 2^full_depth)
    scale = 2 ** (full_depth - 1) 
    scaled_coords = (points_subset + 1.0) * scale  # [batch_size, N, 3]
    
    # 2. 构造查询点 [total_N, 4] (x,y,z,batch_id)
    # 展平并添加batch_id
    total_N = batch_size * N
    flat_coords = scaled_coords.view(total_N, 3)  # [batch_size*N, 3]
    batch_ids = torch.arange(batch_size, device=points_subset.device).repeat_interleave(N).unsqueeze(1)  # [batch_size*N, 1]
    query_points = torch.cat([flat_coords, batch_ids], dim=1)  # [batch_size*N, 4]
    
    # 3. 使用octree.search_xyzb找到对应的voxel位置
    try:
        # 直接搜索对应深度的位置
        indices = octree.search_xyzb(query_points, full_depth, nempty=False)
        
        # 4. 获取对应的xyz坐标
        x, y, z, b = octree.xyzb(full_depth, nempty=False)
        
        # 5. 创建mask
        mask = torch.zeros(batch_size, 8, grid_size, grid_size, grid_size, 
                          dtype=torch.float32, device=points_subset.device)
        
        # 6. 标记找到的位置
        valid_idx = indices >= 0
        if valid_idx.any():
            valid_indices = indices[valid_idx]
            mask_x = x[valid_indices]
            mask_y = y[valid_indices] 
            mask_z = z[valid_indices]
            mask_b = b[valid_indices]
            
            # 在所有8个通道上标记
            mask[mask_b, :, mask_x, mask_y, mask_z] = 1.0
            
        return mask
        
    except:
        print("Octree search failed, using simple coordinate mapping.")

        # 如果octree搜索失败，使用简单的坐标映射
        flat_coords = scaled_coords.view(total_N, 3)
        voxel_coords = flat_coords.floor().long()
        
        # 过滤有效坐标
        valid = torch.all((voxel_coords >= 0) & (voxel_coords < grid_size), dim=1)
        
        mask = torch.zeros(batch_size, 8, grid_size, grid_size, grid_size, 
                          dtype=torch.float32, device=points_subset.device)
        
        if valid.any():
            valid_coords = voxel_coords[valid]
            valid_batch_ids = batch_ids[valid].squeeze(1)  # [num_valid]
            
            # 按batch设置mask
            mask[valid_batch_ids, :, valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = 1.0
            
        return mask

def points_to_node_mask(points_subset, octree, depth, full_depth=2):
    """
    从点云子集生成DualOctree node mask (hierarchical)
    
    Args:
        points_subset (torch.Tensor): 子集点坐标 [batch_size, N, 3], 范围 [-1, 1]  
        octree (Octree): octree结构
        depth (int): depth_stop (e.g. 6)
        full_depth (int): full_depth (e.g. 2)
        
    Returns:
        mask (torch.Tensor): [total_nodes_in_dual_octree, 1]
    """
    batch_size, N, _ = points_subset.shape
    device = points_subset.device
    
    # 1. Calculate offsets and total size
    # We need to replicate DualOctree's indexing logic
    # Indices: [leaf_nodes_full_depth, ..., leaf_nodes_depth-1, nodes_depth]
    
    offsets = {}
    current_offset = 0
    
    # Pre-calculate leaf masks and counts
    leaf_masks = {}
    leaf_cumsums = {}
    
    for d in range(full_depth, depth):
        # Identify leaf nodes: children < 0
        children = octree.children[d]
        is_leaf = children < 0
        leaf_masks[d] = is_leaf
        count = is_leaf.sum().item()
        
        # Mapping from node index to leaf index
        # cumsum gives the index in the leaf array (0-based)
        # We subtract 1 because cumsum starts at 1 for the first True
        leaf_cumsums[d] = torch.cumsum(is_leaf.long(), dim=0) - 1
        
        offsets[d] = current_offset
        current_offset += count
        
    offsets[depth] = current_offset
    total_nodes = current_offset + octree.nnum[depth].item()
    
    mask = torch.zeros(total_nodes, 1, dtype=torch.float32, device=device)
    
    if N == 0:
        return mask
        
    # 2. Process points
    # We process each depth
    for d in range(full_depth, depth + 1):
        scale = 2 ** (d - 1)
        scaled_coords = (points_subset + 1.0) * scale
        # Clamp
        max_val = 2 ** d - 0.001
        scaled_coords = torch.clamp(scaled_coords, 0, max_val)
        
        flat_coords = scaled_coords.view(-1, 3)
        batch_ids = torch.arange(batch_size, device=device).repeat_interleave(N).unsqueeze(1)
        query_points = torch.cat([flat_coords, batch_ids], dim=1)
        
        # Search
        # nempty=False means we search in all nodes (nnum)
        indices = octree.search_xyzb(query_points, d, nempty=False)
        
        valid = indices >= 0
        if not valid.any():
            continue
            
        valid_indices = indices[valid]
        
        if d < depth:
            # For shallower levels, we only care if the point falls into a LEAF node
            is_leaf = leaf_masks[d][valid_indices]
            
            # Filter only those that are leaves
            leaf_hit_indices = valid_indices[is_leaf]
            
            if leaf_hit_indices.numel() > 0:
                # Map to leaf array index
                mapped_indices = leaf_cumsums[d][leaf_hit_indices]
                final_indices = offsets[d] + mapped_indices
                
                # Set mask
                final_indices = torch.unique(final_indices)
                mask[final_indices] = 1.0
                
        else:
            # For depth_stop, we include ALL nodes (leaves and internal)
            final_indices = offsets[d] + valid_indices
            final_indices = torch.unique(final_indices)
            mask[final_indices] = 1.0
            
    return mask


# 使用示例
"""
在OctFusionModel的set_input中使用：

def set_input(self, input=None):
    self.batch_to_cuda(input)
    self.split_small = input['split_small']
    self.octree_in = input['octree_in'] 
    
    # 添加子集mask生成
    if 'points_subset' in input:
        from utils.points_subset_to_split_mask_simple import points_to_split_mask
        
        # input['points_subset'] 形状: [batch_size, N, 3]
        self.subset_mask = points_to_split_mask(
            input['points_subset'], 
            self.octree_in, 
            self.full_depth
        )
        print(f"Generated subset mask: {self.subset_mask.shape}")
    
    # 其他原有代码...
"""