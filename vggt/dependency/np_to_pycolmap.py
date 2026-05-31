# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pycolmap
import os
from .projection import project_3D_points_np
# 👇 新增依赖（可视化 + 核密度估计）
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# 中文/负号兼容配置
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # 英文标题，稳定无乱码
plt.rcParams["axes.unicode_minus"] = False

def batch_np_matrix_to_pycolmap(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    image_size,
    masks=None,
    max_reproj_error=None,
    max_points3D_val=3000,
    shared_camera=False,
    camera_type="PINHOLE",
    extra_params=None,
    min_inlier_per_frame=64,
    points_rgb=None,
    deleted_frames_txt="deleted_frames.txt",
    query_origin_frames=None,  # 轨迹来源查询帧 [P,]
    main_group_inlier_ratio = 0.5
):
    """
    【新版逻辑 - 完全按要求修改】
    1. 逐帧处理，按【查询帧】分成不同分布
    2. 每个分布独立计算：众数±4px 内点
    3. 找到【内点数量最多】的分布 = 主分布
    4. 阈值 = 主分布内点数 × 50%（一半）
    5. 最终保留：主分布 + 内点数 ≥ 阈值的分布
    6. 合并保留组的内点 → 当前帧最终内点
    7. 无合并、无兜底，纯按主分布比例筛选
    """
    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    assert len(intrinsics) == N
    assert len(points3d) == P
    if query_origin_frames is not None:
        assert len(query_origin_frames) == P

    deleted_frame_indices = []
    keep_frame_mask = np.ones(N, dtype=bool)
    final_masks = np.zeros((N, P), dtype=bool)

    if max_reproj_error is not None:
        projected_points_2d, projected_points_cam = project_3D_points_np(points3d, extrinsics, intrinsics)
        projected_diff = np.linalg.norm(projected_points_2d - tracks, axis=-1)
        behind_camera_mask = (projected_points_cam[:, -1] <= 0).reshape(N, P)
        projected_diff[behind_camera_mask] = 1e6

        for fidx in range(N):
            frame_errors = projected_diff[fidx]
            frame_valid_mask = np.ones(P, dtype=bool) if masks is None else masks[fidx]
            frame_valid_mask &= ~behind_camera_mask[fidx]
            valid_indices = np.where(frame_valid_mask)[0]
            
            if len(valid_indices) == 0:
                deleted_frame_indices.append(fidx)
                keep_frame_mask[fidx] = False
                continue

            selected_indices = []
            # ===================== 【你的新核心逻辑：主分布+50%阈值筛选】 =====================
            if query_origin_frames is not None:
                valid_origins = query_origin_frames[valid_indices]
                unique_origins = np.unique(valid_origins)
                
                #  Step1：存储【所有查询帧组】的 内点索引 + 内点数量
                group_results = []  # 每个元素：(inlier_count, inlier_indices)

                # 原代码位置：for origin in unique_origins: 循环内
                for origin in unique_origins:
                    # 取出当前查询帧组的点+误差
                    origin_mask = (valid_origins == origin)
                    group_idx = valid_indices[origin_mask]
                    group_err = frame_errors[group_idx]
                    
                    # ===================== 👇 【新增：打印每一帧每一组的点数量】 =====================
                    group_point_num = len(group_err)
                    # 控制台打印（实时看）
                    # print(f"📊 帧[{fidx}] | 来源查询帧={origin} | 该组轨迹点个数={group_point_num}")
                    # （可选）写入日志文件，永久保存
                    # with open("group_point_count_log.txt", "a", encoding="utf-8") as f:
                    #     f.write(f"帧{fidx} | 查询帧{origin} | 点个数{group_point_num}\n")
                    
                    # ===================== 👇 【新增：防护逻辑，点不足则跳过绘图】 =====================
                    if group_point_num < 2:
                        print(f"⚠️  帧[{fidx}] 查询帧{origin}：点数量<2，跳过绘图+内点计算")
                        group_inlier_idx = np.array([], dtype=int)
                        group_inlier_num = 0
                    else:
                        # 独立算本组众数 ±4px 内点
                        group_mode = compute_error_mode(group_err)
                        # plot_frame_reprojection_error(fidx, origin, group_err, group_mode, max_reproj_error)
                        group_inlier = (group_err >= group_mode - max_reproj_error) & (group_err <= group_mode + max_reproj_error)
                        group_inlier_idx = group_idx[group_inlier]
                        group_inlier_num = len(group_inlier_idx)
                    
                    group_results.append((group_inlier_num, group_inlier_idx))
                
                
                # Step2：找【主分布】（内点数量最多的组）
                group_results.sort(reverse=True, key=lambda x: x[0])  # 按内点数降序
                main_inlier_num, main_inlier_idx = group_results[0]  # 主分布
                threshold_inlier_num = main_inlier_num * main_group_inlier_ratio  # 阈值：主分布的50%
                
                # Step3：筛选组：内点数 ≥ 阈值
                final_groups = [main_inlier_idx]  # 必保留主分布
                for inlier_num, inlier_idx in group_results[1:]:
                    if inlier_num >= threshold_inlier_num:
                        final_groups.append(inlier_idx)
                
                # Step4：合并所有合格组的内点
                for g in final_groups:
                    selected_indices.extend(g)

            # 无分组信息：全局统计（原逻辑不变）
            else:
                valid_err = frame_errors[valid_indices]

                # 👇 【新增：打印全局有效点数量】
                # print(f"📊 帧[{fidx}] | 全局有效轨迹点个数={len(valid_err)}")
                with open("group_point_count_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"帧{fidx} | 全局 | 点个数{len(valid_err)}\n")
        
                mode = compute_error_mode(valid_err)
                # plot_frame_reprojection_error(fidx, origin, group_err, group_mode, max_reproj_error)
                inlier = (valid_err >= mode - max_reproj_error) & (valid_err <= mode + max_reproj_error)
                selected_indices = valid_indices[inlier]

            # 内点不足删帧（原逻辑）
            if len(selected_indices) < min_inlier_per_frame:
                deleted_frame_indices.append(fidx)
                keep_frame_mask[fidx] = False
                continue
            final_masks[fidx, selected_indices] = True

        # 过滤帧（原逻辑）
        masks = final_masks[keep_frame_mask]
        extrinsics = extrinsics[keep_frame_mask]
        intrinsics = intrinsics[keep_frame_mask]
        tracks = tracks[keep_frame_mask]
        N = len(extrinsics)
        if N == 0:
            print("所有帧都被删除，无法进行BA")
            with open(deleted_frames_txt, "w") as f:
                f.write("\n".join(map(str, deleted_frame_indices)))
            return None, None

    else:
        if masks is None:
            masks = np.ones((N, P), dtype=bool)

    # 保存删帧记录 + BA后续逻辑（完全不变）
    if deleted_frame_indices:
        with open(deleted_frames_txt, "w") as f:
            f.write("\n".join(map(str, deleted_frame_indices)))
        print(f"删除 {len(deleted_frame_indices)} 帧：{deleted_frame_indices}")

    inliers_per_frame = masks.sum(1)
    if inliers_per_frame.min() < min_inlier_per_frame:
        print(f"最小内点{inliers_per_frame.min()} < {min_inlier_per_frame}，跳过BA")
        return None, None

    reconstruction = pycolmap.Reconstruction()
    valid_mask = masks.sum(0) >= 2
    valid_idx = np.nonzero(valid_mask)[0]
    for vidx in valid_idx:
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx] if points_rgb is not None else [0,0,0])

    num_points3D = len(valid_idx)
    camera = None
    for fidx in range(N):
        if camera is None or not shared_camera:
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params)
            camera = pycolmap.Camera(model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=fidx+1)
            reconstruction.add_camera(camera)
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(extrinsics[fidx][:3,:3]), extrinsics[fidx][:3,3])
        image = pycolmap.Image(id=fidx+1, name=f"image_{fidx+1}", camera_id=camera.camera_id, cam_from_world=cam_from_world)
        points2D_list = []
        pid = 0
        for point3D_id in range(1, num_points3D+1):
            ot = valid_idx[point3D_id-1]
            if masks[fidx, ot]:
                points2D_list.append(pycolmap.Point2D(tracks[fidx, ot], point3D_id))
                reconstruction.points3D[point3D_id].track.add_element(fidx+1, pid)
                pid +=1
        image.points2D = pycolmap.ListPoint2D(points2D_list)
        image.registered = True
        reconstruction.add_image(image)

    return reconstruction, valid_mask

def pycolmap_to_batch_np_matrix(reconstruction, device="cpu", camera_type="SIMPLE_PINHOLE"):
    """
    Convert a PyCOLMAP Reconstruction Object to batched NumPy arrays.

    Args:
        reconstruction (pycolmap.Reconstruction): The reconstruction object from PyCOLMAP.
        device (str): Ignored in NumPy version (kept for API compatibility).
        camera_type (str): The type of camera model used (default: "SIMPLE_PINHOLE").

    Returns:
        tuple: A tuple containing points3D, extrinsics, intrinsics, and optionally extra_params.
    """

    num_images = len(reconstruction.images)
    max_points3D_id = max(reconstruction.point3D_ids())
    points3D = np.zeros((max_points3D_id, 3))

    for point3D_id in reconstruction.points3D:
        points3D[point3D_id - 1] = reconstruction.points3D[point3D_id].xyz

    extrinsics = []
    intrinsics = []

    extra_params = [] if camera_type == "SIMPLE_RADIAL" else None

    for i in range(num_images):
        # Extract and append extrinsics
        pyimg = reconstruction.images[i + 1]
        pycam = reconstruction.cameras[pyimg.camera_id]
        matrix = pyimg.cam_from_world.matrix()
        extrinsics.append(matrix)

        # Extract and append intrinsics
        calibration_matrix = pycam.calibration_matrix()
        intrinsics.append(calibration_matrix)

        if camera_type == "SIMPLE_RADIAL":
            extra_params.append(pycam.params[-1])

    # Convert lists to NumPy arrays instead of torch tensors
    extrinsics = np.stack(extrinsics)
    intrinsics = np.stack(intrinsics)

    if camera_type == "SIMPLE_RADIAL":
        extra_params = np.stack(extra_params)
        extra_params = extra_params[:, None]

    return points3D, extrinsics, intrinsics, extra_params


def batch_np_matrix_to_pycolmap_wo_track(
    points3d,
    points_xyf,
    points_rgb,
    extrinsics,
    intrinsics,
    image_size,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Different from batch_np_matrix_to_pycolmap, this function does not use tracks.

    It saves points3d to colmap reconstruction format only to serve as init for Gaussians or other nvs methods.

    Do NOT use this for BA.
    """
    # points3d: Px3
    # points_xyf: Px3, with x, y coordinates and frame indices
    # points_rgb: Px3, rgb colors
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N = len(extrinsics)
    P = len(points3d)

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    for vidx in range(P):
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx])

    camera = None
    # frame idx
    for fidx in range(N):
        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type)

            camera = pycolmap.Camera(
                model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=fidx + 1
            )

            # add camera
            reconstruction.add_camera(camera)

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        image = pycolmap.Image(
            id=fidx + 1, name=f"image_{fidx + 1}", camera_id=camera.camera_id, cam_from_world=cam_from_world
        )

        points2D_list = []

        point2D_idx = 0

        points_belong_to_fidx = points_xyf[:, 2].astype(np.int32) == fidx
        points_belong_to_fidx = np.nonzero(points_belong_to_fidx)[0]

        for point3D_batch_idx in points_belong_to_fidx:
            point3D_id = point3D_batch_idx + 1
            point2D_xyf = points_xyf[point3D_batch_idx]
            point2D_xy = point2D_xyf[:2]
            points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

            # add element
            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
        except:
            print(f"frame {fidx + 1} does not have any points")
            image.registered = False

        # add image
        reconstruction.add_image(image)

    return reconstruction


def _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    """
    Helper function to get camera parameters based on camera type.

    Args:
        fidx: Frame index
        intrinsics: Camera intrinsic parameters
        camera_type: Type of camera model
        extra_params: Additional parameters for certain camera types

    Returns:
        pycolmap_intri: NumPy array of camera parameters
    """
    if camera_type == "PINHOLE":
        pycolmap_intri = np.array(
            [intrinsics[fidx][0, 0], intrinsics[fidx][1, 1], intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]])
    elif camera_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL is not supported yet")
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2], extra_params[fidx][0]])
    else:
        raise ValueError(f"Camera type {camera_type} is not supported yet")

    return pycolmap_intri


# ==============================================================================
# 工具1：计算重投影误差分布的众数（概率密度峰值）
# ==============================================================================
def compute_error_mode(errors: np.ndarray, bandwidth: float = 0.3) -> float:
    """
    计算误差分布的众数（概率密度最高的误差值）
    点过少时自动返回中位数，保证鲁棒性
    """
    # 👇 【强防护】小于2个点，直接返回中位数，绝不进KDE
    if len(errors) < 2:
        return float(np.median(errors)) if len(errors) > 0 else 0.0
    if len(errors) < 10:
        return float(np.median(errors))
    # 核密度估计找峰值
    kde = gaussian_kde(errors, bw_method=bandwidth)
    x_eval = np.linspace(errors.min(), errors.max(), 200)
    kde_vals = kde(x_eval)
    return float(x_eval[np.argmax(kde_vals)])

# ==============================================================================
# 👇 封装好的独立函数：绘制单帧重投影误差分布（无侵入、不影响原逻辑）
# ==============================================================================
def plot_frame_reprojection_error(
    fidx: int,              # 帧号
    origin: int,            # 【新增】查询帧来源号
    valid_errors: np.ndarray,
    mode: float,
    window_px: float,
    save_dir: str = "/root/autodl-tmp/vggt/check_reproj"
):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图 + 密度曲线
    plt.hist(valid_errors, bins=50, color="#4472c4", alpha=0.7, edgecolor="black", density=True)
    kde = gaussian_kde(valid_errors, bw_method=0.3)
    x_eval = np.linspace(0, 100, 200)  # 固定 0~100 计算密度
    plt.plot(x_eval, kde(x_eval), 'r-', linewidth=2, label="Probability Density")
    
    # 绘制众数与阈值线
    plt.axvline(mode, color="green", linewidth=3, label=f"Mode = {mode:.2f}px")
    plt.axvline(mode - window_px, color="orange", linestyle="--", label=f"Mode-{window_px}px")
    plt.axvline(mode + window_px, color="orange", linestyle="--", label=f"Mode+{window_px}px")
    
    # ==============================================
    # 🔥 核心修复：强制固定 X 轴范围 0 ~ 100px
    # ==============================================
    plt.xlim(0, 100)
    
    plt.xlabel("Reprojection Error (px)")
    plt.ylabel("Density")
    plt.title(f"Frame {fidx} | Origin {origin} | Reprojection Error (0-100px fixed)")
    plt.legend()
    plt.grid(alpha=0.3)

    save_path = os.path.join(save_dir, f"frame_{fidx}_origin_{origin}_reproj.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()