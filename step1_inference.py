# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
import trimesh
import utils.opt as opt_utils
import utils.colmap as colmap_utils
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from utils.metric_torch import evaluate_auc, evaluate_pcd, write_evaluation_results

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_ratio
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track

torch._dynamo.config.accumulated_cache_size_limit = 512

def run_VGGT(images, device, dtype, chunk_size):
    # images: [B, 3, H, W]
    local_model_path = "/root/autodl-tmp/model.pt"
    model = VGGT(chunk_size=chunk_size)
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()

    
    model = model.to(device).to(dtype)
    model.track_head = None  # we do not need tracking head for reconstruction
    print(f"Model loaded")

    with torch.no_grad():
        predictions = model(images.to(device, dtype), verbose=True)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'], images.shape[-2:])
        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy()
        depth_map = predictions['depth'].squeeze(0).cpu().numpy()
        depth_conf = predictions['depth_conf'].squeeze(0).cpu().numpy()
    
    return extrinsic, intrinsic, depth_map, depth_conf

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--chunk_size", type=int, default=256, help="Chunk size for frame-wise operation in VGGT")
    parser.add_argument("--total_frame_num", type=int, default=None, help="Number of frames to reconstruct")
    return parser.parse_args()

def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype —— 修复：CPU环境不调用CUDA API，避免崩溃
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    ############################################## 第一步：加载图片与预处理 #############################################
    image_dir = os.path.join(args.scene_dir, "images")
    # 修复：避免os.listdir统计非图像文件，基于glob结果定数量，防止切片异常
    image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))
    if not image_path_list:
        raise ValueError(f"No images found in {image_dir}")
    if args.total_frame_num is None:
        args.total_frame_num = len(image_path_list)
    # 修复：total_frame_num非负+超界修正，避免取空列表
    args.total_frame_num = max(0, min(args.total_frame_num, len(image_path_list)))
    image_path_list = image_path_list[:args.total_frame_num]

    inverse_idx = list(range(len(image_path_list)))
    base_image_path_list = [os.path.basename(path) for path in image_path_list]
    base_image_path_list_inv = [base_image_path_list[i] for i in inverse_idx]

    # 你的原始代码，无任何修改
    images, original_coords = load_and_preprocess_images_ratio(image_path_list, 518)
    # 修复：CUDA Tensor转CPU标量，避免后续计算/转numpy崩溃
    ori_w = original_coords[0][4].cpu().item()  # original width
    ori_h = original_coords[0][5].cpu().item()  # original height
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # 修复：CPU环境不调用CUDA显存重置，避免崩溃
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = datetime.now()

    ################################################## 第二步：VGGT 推理 ################################################
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(images, device, dtype, args.chunk_size)

    ################################################## 第三步：保存结果 ################################################
    # 你的原始逻辑，一字未改 —— 完全保留你定义的所有变量和计算方式
    vggt_resolution = (depth_map.shape[1], depth_map.shape[2])
    if ori_w > ori_h:
        scale = 1024.0 / ori_w
        img_load_resolution = (ori_h * scale, 1024.0)
    else:
        scale = 1024.0 / ori_h
        img_load_resolution = (1024.0, ori_w * scale)
    
    # ########### 你的save_dict，完全无修改，一字未动 ###########
    save_dict = {
        "extrinsic": extrinsic.astype(np.float32),
        "intrinsic": intrinsic.astype(np.float32),
        "depth_map": depth_map.astype(np.float32),
        "depth_conf": depth_conf.astype(np.float32),
        "vggt_resolution": np.array(vggt_resolution, dtype=np.float32),
        "img_load_resolution": np.array(img_load_resolution, dtype=np.float32)
    }
    # ###########################################################

    # 修复：创建输出目录，避免保存时路径不存在报错
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "vggt_x_for_ba.npy"), save_dict)
    print(f"结果已保存至：{os.path.join(args.output_dir, 'vggt_x_for_ba.npy')}")
    print(f"vggt_resolution: {vggt_resolution}, img_load_resolution: {img_load_resolution}")

    return True

if __name__ == "__main__":
    args = parse_args()
    demo_fn(args)