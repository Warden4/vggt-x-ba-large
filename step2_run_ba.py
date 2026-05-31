from PIL import Image
import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap

from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap
from vggt.utils.load_fn import load_and_preprocess_images_square, load_and_preprocess_images_proportional

# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT BA Optimization with Chunk Data")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images (for track prediction)")
    parser.add_argument("--chunk_path", type=str, required=True, help="Path to chunk_for_ba.npy file (contains extrinsic/intrinsic/depth data)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=True, help="Use BA for reconstruction (default: True)")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=4, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=True, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.95, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=10, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=False, help="Use fine tracking (slower but more accurate)"
    )
    return parser.parse_args()


def load_chunk_for_ba(chunk_path):
    """
    加载chunk_for_ba.npy文件，校验数据格式并返回关键参数
    """
    # 加载文件
    if not os.path.exists(chunk_path):
        raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
    
    chunk_data = np.load(chunk_path, allow_pickle=True).item()
    
    # 校验必填字段
    required_fields = ["extrinsic", "intrinsic", "depth_map", "depth_conf"]
    missing_fields = [f for f in required_fields if f not in chunk_data]
    if missing_fields:
        raise ValueError(f"Chunk file missing required fields: {missing_fields}")
    
    # 提取并校验数据格式
    extrinsic = chunk_data["extrinsic"]  # [N, 3, 4] (W2C, OpenCV)
    intrinsic = chunk_data["intrinsic"]  # [N, 3, 3] (518分辨率)
    depth_map = chunk_data["depth_map"]  # [N, H, W, 1]
    depth_conf = chunk_data["depth_conf"]  # [N, H, W]
    vggt_resolution = chunk_data["vggt_resolution"]  # (154, 518)
    img_load_resolution = chunk_data["img_load_resolution"]  # (304, 1024)

    # 拆分分辨率的高/宽（保留）
    vggt_h, vggt_w = vggt_resolution
    load_h, load_w = img_load_resolution
    
    # 维度校验
    if extrinsic.ndim != 3 or extrinsic.shape[1:] != (3, 4):
        raise ValueError(f"Invalid extrinsic shape: {extrinsic.shape}, expected [N, 3, 4]")
    if intrinsic.ndim != 3 or intrinsic.shape[1:] != (3, 3):
        raise ValueError(f"Invalid intrinsic shape: {intrinsic.shape}, expected [N, 3, 3]")
    if depth_map.ndim != 4 or depth_map.shape[1:3] != (vggt_h, vggt_w):
        raise ValueError(f"Invalid depth_map shape: {depth_map.shape}, expected [N, {vggt_h}, {vggt_w}, 1]")
    
    print(f"✅ Chunk数据加载成功 | 帧数: {len(extrinsic)} | 分辨率: {vggt_resolution}")
    print(f"  - extrinsic: {extrinsic.shape}")
    print(f"  - intrinsic: {intrinsic.shape}")
    print(f"  - depth_map: {depth_map.shape}")
    print(f"  - depth_conf: {depth_conf.shape}")
    
    return extrinsic, intrinsic, depth_map, depth_conf, vggt_resolution, img_load_resolution


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

    # Set device and dtype (仅用于轨迹预测，保留原逻辑)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    ############################################ 第一步：加载Chunk数据（替代VGGT推理） ####################################################
    print("\n📌 加载chunk_for_ba.npy数据（替代VGGT模型推理）")
    extrinsic, intrinsic, depth_map, depth_conf, vggt_resolution, img_load_resolution = load_chunk_for_ba(args.chunk_path)
    total_frames = len(extrinsic)

    ############################################ 第二步：加载图片（用于轨迹预测） ##################################################
    print("\n📌 加载场景图片（用于轨迹预测）")
    # 加载图片路径（需和chunk数据的帧数匹配）
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    # 按数字排序图片（保证和chunk数据帧顺序一致）
    def extract_num(path):
        fname = os.path.basename(path)
        num = ''.join(filter(str.isdigit, fname))
        return int(num) if num else 0
    image_path_list = sorted(image_path_list, key=extract_num)
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # 校验图片数量和chunk帧数匹配
    if len(image_path_list) != total_frames:
        raise ValueError(f"Image count ({len(image_path_list)}) != Chunk frame count ({total_frames})")
    
    # 加载图片（仅用于轨迹预测，保留原预处理逻辑）
    images, original_coords = load_and_preprocess_images_proportional(image_path_list, 1024)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    ############################################ 第三步：3D点云反投影 ##################################################
    print("\n📌 从深度图反投影生成3D点云")
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    print(f"3D点云生成完成 | 形状: {points_3d.shape}")

    ############################################ 第四步：内参缩放（适配1024分辨率） ##################################################
    # 原逻辑：将518分辨率的内参缩放到img_load_resolution（1024）
    shared_camera = args.shared_camera
    intrinsic_scaled = intrinsic.copy()
    # 拆分高宽
    vggt_h, vggt_w = vggt_resolution  
    load_h, load_w = img_load_resolution

    # 分别计算宽、高缩放比
    scale_w = load_w / vggt_w
    scale_h = load_h / vggt_h

    # 宽度维度参数（fx, cx）乘scale_w；高度维度参数（fy, cy）乘scale_h
    intrinsic_scaled[:, 0, :] *= scale_w  # 缩放 fx, cx
    intrinsic_scaled[:, 1, :] *= scale_h  # 缩放 fy, cy

    print(f"📌 内参缩放 | {vggt_h}x{vggt_w} → {load_h}x{load_w} | 缩放比例: {scale_w:.4f}, {scale_h:.4f}")

    ############################################ 第五步：VGGSfM 轨迹预测 ##################################################
    print("\n📌 执行VGGSfM轨迹预测")
    print(f"图像尺寸：{images.shape}")
    print(f"置信度尺寸：{depth_conf.shape}")
    print(f"点云尺寸：{points_3d.shape}")
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predicting Tracks
        pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb, query_origin_frames = predict_tracks(
            images,
            conf=depth_conf,
            points_3d=points_3d,
            masks=None,
            max_query_pts=args.max_query_pts,
            query_frame_num=args.query_frame_num,
            keypoint_extractor="aliked+sp",
            fine_tracking=args.fine_tracking
        )

        torch.cuda.empty_cache()

    track_mask = pred_vis_scores > args.vis_thresh
    print(f"轨迹预测完成 | 轨迹掩码形状: {track_mask.shape}")
    
    """
    # 新增保存轨迹查看（修复后）
    save_dir = "/root/autodl-tmp/vggt/track_drone"  # 统一保存路径，方便后续查找
    os.makedirs(save_dir, exist_ok=True)

    # ========== 关键修复：先验证数据类型，再处理 ==========
    try:
        # 1. 处理并保存轨迹相关数据（原有逻辑保留）
        if isinstance(pred_tracks, torch.Tensor):
            pred_tracks_np = pred_tracks.detach().cpu().numpy()
            pred_vis_scores_np = pred_vis_scores.detach().cpu().numpy()
            track_mask_np = track_mask.detach().cpu().numpy() if isinstance(track_mask, torch.Tensor) else track_mask
            # 处理3D点和RGB（核心：保存VGGSfM输出的原始3D点）
            points_3d_np = points_3d.detach().cpu().numpy() if isinstance(points_3d, torch.Tensor) else points_3d
            points_rgb_np = points_rgb.detach().cpu().numpy() if (points_rgb is not None and isinstance(points_rgb, torch.Tensor)) else points_rgb
        else:
            pred_tracks_np = pred_tracks
            pred_vis_scores_np = pred_vis_scores
            track_mask_np = track_mask
            points_3d_np = points_3d
            points_rgb_np = points_rgb

        # 2. 保存轨迹基础数据（原有）
        np.save(os.path.join(save_dir, "pred_tracks.npy"), pred_tracks_np)
        np.save(os.path.join(save_dir, "pred_vis_scores.npy"), pred_vis_scores_np)
        np.save(os.path.join(save_dir, "track_mask.npy"), track_mask_np)

        # 3. 核心新增：保存VGGSfM提前建立的原始3D点（关键！）
        # 保存完整原始3D点 + RGB + 轨迹索引
        np.savez(
            os.path.join(save_dir, "vggsfm_raw_3d_points.npz"),
            points_3d=points_3d_np,          # VGGSfM输出的原始3D点 [P, 3]
            points_rgb=points_rgb_np if points_rgb_np is not None else np.zeros_like(points_3d_np),  # 3D点RGB
            track_indices=np.arange(len(points_3d_np)),  # 轨迹索引（与3D点一一对应）
            pred_tracks_shape=pred_tracks_np.shape,      # 轨迹形状（验证维度匹配）
            num_3d_points=len(points_3d_np)              # 3D点总数
        )

        # 4. 保存3D点-轨迹的绑定关系（验证同一3D点对应多帧2D轨迹）
        # 构建：每个3D点在哪些帧有可见轨迹（track_mask=True）
        point_frame_binding = {}
        num_frames, num_points = pred_tracks_np.shape[:2]
        for p_idx in range(min(1000, num_points)):  # 可选：限制数量避免文件过大，也可全部保存
            # 找到该3D点在哪些帧有可见轨迹
            valid_frames = np.nonzero(track_mask_np[:, p_idx])[0]
            if len(valid_frames) > 0:
                point_frame_binding[p_idx] = {
                    "valid_frames": valid_frames,                  # 可见帧索引
                    "frame_2d_coords": pred_tracks_np[valid_frames, p_idx],  # 对应帧的2D轨迹坐标
                    "vis_scores": pred_vis_scores_np[valid_frames, p_idx]    # 可见度分数
                }
        np.savez(
            os.path.join(save_dir, "vggsfm_track_3d_binding.npz"),
            **point_frame_binding
        )

        # 5. 打印保存信息（新增3D点相关）
        print(f"\n✅ 数据保存完成！路径：{save_dir}")
        print(f"   - 轨迹数据：pred_tracks.shape={pred_tracks_np.shape}")
        print(f"   - 可见度分数：pred_vis_scores.shape={pred_vis_scores_np.shape}")
        print(f"   - VGGSfM原始3D点：points_3d.shape={points_3d_np.shape}（总数：{len(points_3d_np)}）")
        print(f"   - 有可见轨迹的3D点数量：{len(point_frame_binding)}")

    except Exception as e:
        # 捕获异常，避免程序中断，同时打印错误信息
        print(f"\n❌ 保存轨迹/3D点数据失败: {e}")
        import traceback
        traceback.print_exc()
    """
    ################################################ 第六步：全局 BA 优化 ################################################
    print("\n📌 执行全局BA优化")
    img_load_resolution_tuple = (load_w, load_h)
    img_load_resolution = np.array(img_load_resolution_tuple, dtype=np.int32)  # 注意顺序 (W, H)
    # TODO: radial distortion, iterative BA, masks
    reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
        points_3d,
        extrinsic,
        intrinsic_scaled,  # 使用缩放后的内参
        pred_tracks,
        image_size=img_load_resolution,
        masks=track_mask,
        max_reproj_error=args.max_reproj_error,
        shared_camera=shared_camera,
        camera_type=args.camera_type,
        points_rgb=points_rgb,
        query_origin_frames=query_origin_frames
    )

    if reconstruction is None:
        raise ValueError("No reconstruction can be built with BA")

    # Bundle Adjustment
    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, ba_options)
    print("✅ BA优化完成")

    reconstruction_resolution = img_load_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    ################################################ 第七步：保存结果 ################################################
    print(f"\n📌 保存重建结果到 {args.scene_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(args.scene_dir, "sparse/points.ply"))
    print("✅ 所有结果保存完成！")

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            pred_params = copy.deepcopy(pycamera.params)
            real_image_size = original_coords[pyimageid - 1, -2:]  # 原始分辨率 (W, H)
            
            # ✅ 处理img_size：如果是单数值（兼容旧逻辑），转正方形；如果是元组，取宽高
            if isinstance(img_size, (int, float)):
                load_w = load_h = img_size
            else:
                load_h, load_w = img_size  # 匹配你img_load_resolution的格式 (H, W)
            
            # ✅ 分维度计算缩放比（核心）
            scale_w = real_image_size[0] / load_w  # 原始宽 / 加载宽
            scale_h = real_image_size[1] / load_h  # 原始高 / 加载高

            # ✅ 分维度缩放相机参数（适配SIMPLE_PINHOLE/PINHOLE）
            if len(pred_params) == 3:  # SIMPLE_PINHOLE: fx, cx, cy
                pred_params[0] *= scale_w  # fx（宽度维度）
                pred_params[1] *= scale_w  # cx（宽度维度）
                pred_params[2] *= scale_h  # cy（高度维度）
            elif len(pred_params) == 4:  # PINHOLE: fx, fy, cx, cy
                pred_params[0] *= scale_w  # fx
                pred_params[1] *= scale_h  # fy
                pred_params[2] *= scale_w  # cx
                pred_params[3] *= scale_h  # cy

            # ✅ 修正主点到原始图像中心
            real_pp_w = real_image_size[0] / 2
            real_pp_h = real_image_size[1] / 2
            if len(pred_params) == 3:
                pred_params[1] = real_pp_w
                pred_params[2] = real_pp_h
            elif len(pred_params) == 4:
                pred_params[2] = real_pp_w
                pred_params[3] = real_pp_h

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            top_left = original_coords[pyimageid - 1, :2]
            # ✅ 处理img_size为矩形
            if isinstance(img_size, (int, float)):
                load_w = load_h = img_size
            else:
                load_h, load_w = img_size
            real_image_size = original_coords[pyimageid - 1, -2:]
            scale_w = real_image_size[0] / load_w
            scale_h = real_image_size[1] / load_h

            for point2D in pyimage.points2D:
                x = (point2D.xy[0] - top_left[0]) * scale_w
                y = (point2D.xy[1] - top_left[1]) * scale_h
                point2D.xy = (x, y)

        if shared_camera:
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)
