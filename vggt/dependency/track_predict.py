# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from .vggsfm_utils import *

def predict_tracks(
    images,
    conf=None,
    points_3d=None,
    masks=None,
    max_query_pts=2048,
    query_frame_num=5,
    keypoint_extractor="aliked+sp",
    max_points_num=163840,
    fine_tracking=True,
    complete_non_vis=True,
):
    """
    Returns:
        pred_tracks: (N, P, 2)  总轨迹数P
        pred_vis_scores: (N, P)
        query_origin_frames: (P,) → 【全真实来源帧】，长度=P，无-1，永不报错
    """
    device = images.device
    dtype = images.dtype
    tracker = build_vggsfm_tracker().to(device, dtype)

    # 生成查询帧序列
    query_frame_indexes = generate_rank_by_dino(images, query_frame_num=query_frame_num, device=device)
    # query_frame_indexes = generate_uniform_queries(images.shape[0], query_num=query_frame_num, stride=None)
    
    if 0 in query_frame_indexes:
        query_frame_indexes.remove(0)
    query_frame_indexes = [0, *query_frame_indexes]

    keypoint_extractors = initialize_feature_extractors(
        max_query_pts, extractor_method=keypoint_extractor, device=device
    )

    # 轨迹容器 + 来源标签容器
    pred_tracks = []
    pred_vis_scores = []
    pred_confs = []
    pred_points_3d = []
    pred_colors = []
    query_origin_frames = []

    fmaps_for_tracker = tracker.process_images_to_fmaps(images)
    if fine_tracking:
        print("For faster inference, consider disabling fine_tracking")

    # ========== 工具函数：添加轨迹 + 同步打【真实来源标签】 ==========
    def add_track_with_origin(query_idx, pred_track, pred_vis, pred_conf, pred_point_3d, pred_color):
        pred_tracks.append(pred_track)
        pred_vis_scores.append(pred_vis)
        pred_confs.append(pred_conf)
        pred_points_3d.append(pred_point_3d)
        pred_colors.append(pred_color)
        # 【真标签】用传入的query_idx，无-1
        num_tracks = pred_track.shape[1]
        query_origin_frames.append(np.full((num_tracks,), query_idx, dtype=np.int32))

    # 1. 初始查询帧生成轨迹（打真标签）
    for query_index in query_frame_indexes:
        print(f"Predicting tracks for query frame {query_index}")
        pred_track, pred_vis, pred_conf, pred_point_3d, pred_color = _forward_on_query(
            query_index, images, conf, points_3d, fmaps_for_tracker,
            keypoint_extractors, tracker, max_points_num, fine_tracking, device
        )
        add_track_with_origin(query_index, pred_track, pred_vis, pred_conf, pred_point_3d, pred_color)

    # 2. 轨迹增强（补全低可见帧）→ ✅ 接收【真实来源帧】，打真标签
    if complete_non_vis:
        print("Augmenting tracks for non-visible frames...")
        # ========== 核心修复：接收增强返回的「新增轨迹+对应真实来源帧」 ==========
        pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors, added_origins = _augment_non_visible_frames(
            pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors,
            images, conf, points_3d, fmaps_for_tracker, keypoint_extractors,
            tracker, max_points_num, fine_tracking, device=device
        )
        # ========== 给增强轨迹打【真标签】（完全匹配，无-1） ==========
        for query_idx, track in zip(added_origins, pred_tracks[-len(added_origins):]):
            num_tracks = track.shape[1]
            query_origin_frames.append(np.full((num_tracks,), query_idx, dtype=np.int32))

    # 最终拼接（长度100%匹配，全真实标签）
    pred_tracks = np.concatenate(pred_tracks, axis=1)
    pred_vis_scores = np.concatenate(pred_vis_scores, axis=1)
    pred_confs = np.concatenate(pred_confs, axis=0) if pred_confs else None
    pred_points_3d = np.concatenate(pred_points_3d, axis=0) if pred_points_3d else None
    pred_colors = np.concatenate(pred_colors, axis=0) if pred_colors else None
    query_origin_frames = np.concatenate(query_origin_frames, axis=0)

    # 可选校验（永远打印：匹配成功）
    # print(f"✅ 校验：总轨迹数={pred_tracks.shape[1]}, 标签数={len(query_origin_frames)}")
    # print(f"✅ 来源帧取值：{np.unique(query_origin_frames)}（全是真实查询帧，无-1）")

    return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors, query_origin_frames


# ===================== _forward_on_query 完全不变（你原来的代码） =====================
def _forward_on_query(
    query_index,
    images,
    conf,
    points_3d,
    fmaps_for_tracker,
    keypoint_extractors,
    tracker,
    max_points_num,
    fine_tracking,
    device,
):
    frame_num, _, img_h, img_w = images.shape
    query_image = images[query_index]
    query_points = extract_keypoints(query_image, keypoint_extractors, round_keypoints=False)
    query_points = query_points[:, torch.randperm(query_points.shape[1], device=device)]

    query_points_long = query_points.squeeze(0).round().long()
    query_points_long[:, 0] = torch.clamp(query_points_long[:, 0], 0, img_w - 1)
    query_points_long[:, 1] = torch.clamp(query_points_long[:, 1], 0, img_h - 1)
    
    pred_color = images[query_index][:, query_points_long[:, 1], query_points_long[:, 0]]
    pred_color = (pred_color.permute(1, 0).cpu().numpy() * 255).astype(np.uint8)

    if (conf is not None) and (points_3d is not None):
        conf_h, conf_w = conf.shape[-2:]
        points_h, points_w = points_3d.shape[-3:-1]
        assert (conf_h, conf_w) == (points_h, points_w)
        
        scale_w = conf_w / img_w
        scale_h = conf_h / img_h

        query_points_squeeze = query_points.squeeze(0)
        query_points_scaled_x = (query_points_squeeze[:, 0] * scale_w).round().long()
        query_points_scaled_y = (query_points_squeeze[:, 1] * scale_h).round().long()
        query_points_scaled = torch.stack([query_points_scaled_x, query_points_scaled_y], dim=1).cpu().numpy()
        query_points_scaled[:, 0] = np.clip(query_points_scaled[:, 0], 0, conf_w - 1)
        query_points_scaled[:, 1] = np.clip(query_points_scaled[:, 1], 0, conf_h - 1)

        pred_conf = conf[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]
        pred_point_3d = points_3d[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]

        valid_mask = pred_conf > 1.2
        if valid_mask.sum() > 512:
            query_points = query_points[:, valid_mask]
            pred_conf = pred_conf[valid_mask]
            pred_point_3d = pred_point_3d[valid_mask]
            pred_color = pred_color[valid_mask]
    else:
        pred_conf = None
        pred_point_3d = None

    reorder_index = calculate_index_mappings(query_index, frame_num, device=device)
    images_feed, fmaps_feed = switch_tensor_order([images, fmaps_for_tracker], reorder_index, dim=0)
    images_feed = images_feed[None]
    fmaps_feed = fmaps_feed[None]

    all_points_num = images_feed.shape[1] * query_points.shape[1]
    if all_points_num > max_points_num:
        num_splits = (all_points_num + max_points_num - 1) // max_points_num
        query_points = torch.chunk(query_points, num_splits, dim=1)
    else:
        query_points = [query_points]

    pred_track, pred_vis, _ = predict_tracks_in_chunks(
        tracker, images_feed, query_points, fmaps_feed, fine_tracking=fine_tracking
    )

    pred_track, pred_vis = switch_tensor_order([pred_track, pred_vis], reorder_index, dim=1)
    pred_track = pred_track.squeeze(0).float().cpu().numpy()
    pred_vis = pred_vis.squeeze(0).float().cpu().numpy()

    return pred_track, pred_vis, pred_conf, pred_point_3d, pred_color


def _forward_on_query(
    query_index,
    images,
    conf,
    points_3d,
    fmaps_for_tracker,
    keypoint_extractors,
    tracker,
    max_points_num,
    fine_tracking,
    device,
):
    """
    Process a single query frame for track prediction.
    ✅ 适配任意长宽比矩形图片
    ✅ 分维度缩放坐标（x/y分别用宽/高比例）

    Args:
        query_index: Index of the query frame
        images: Tensor of shape [S, 3, H, W] containing the input images (矩形分辨率)
        conf: Confidence tensor of shape [S, H, W] (vggt_resolution)
        points_3d: 3D points tensor of shape [S, H, W, 3] (vggt_resolution)
        fmaps_for_tracker: Feature maps for the tracker
        keypoint_extractors: Initialized feature extractors
        tracker: VGG-SFM tracker
        max_points_num: Maximum number of points to process at once
        fine_tracking: Whether to use fine tracking
        device: Device to use for computation

    Returns:
        pred_track: Predicted tracks
        pred_vis: Visibility scores for the tracks
        pred_conf: Confidence scores for the tracks
        pred_point_3d: 3D points for the tracks
        pred_color: Point colors for the tracks (0, 255)
    """
    frame_num, _, img_h, img_w = images.shape  # 图片分辨率（img_load_resolution: H, W）

    query_image = images[query_index]
    query_points = extract_keypoints(query_image, keypoint_extractors, round_keypoints=False)
    query_points = query_points[:, torch.randperm(query_points.shape[1], device=device)]

    # Extract the color at the keypoint locations
    query_points_long = query_points.squeeze(0).round().long()
    # 保护逻辑：防止坐标越界（矩形图片可能出现）
    query_points_long[:, 0] = torch.clamp(query_points_long[:, 0], 0, img_w - 1)
    query_points_long[:, 1] = torch.clamp(query_points_long[:, 1], 0, img_h - 1)
    
    pred_color = images[query_index][:, query_points_long[:, 1], query_points_long[:, 0]]
    pred_color = (pred_color.permute(1, 0).cpu().numpy() * 255).astype(np.uint8)

    # Query the confidence and points_3d at the keypoint locations
    if (conf is not None) and (points_3d is not None):
        # ========== 关键修改1：移除正方形强制断言 ==========
        # 原代码：assert height == width / assert conf.shape[-2] == conf.shape[-1]
        # 替换为：校验conf和points_3d的维度匹配（矩形兼容）
        conf_h, conf_w = conf.shape[-2:]  # conf的分辨率（vggt_resolution: H, W）
        points_h, points_w = points_3d.shape[-3:-1]
        assert (conf_h, conf_w) == (points_h, points_w), \
            f"conf分辨率({conf_h},{conf_w})与points_3d分辨率({points_h},{points_w})不匹配"
        
        # ========== 关键修改2：分维度计算缩放比例（适配矩形） ==========
        # 原代码：scale = conf.shape[-1] / width （单比例，仅支持正方形）
        # 新逻辑：x（宽度）用scale_w，y（高度）用scale_h
        scale_w = conf_w / img_w  # 宽度缩放比例（conf_w / 图片宽度）
        scale_h = conf_h / img_h  # 高度缩放比例（conf_h / 图片高度）

        # ========== 关键修改3：分维度缩放坐标 ==========
        query_points_squeeze = query_points.squeeze(0)
        # x坐标（宽度维度）用scale_w缩放，y坐标（高度维度）用scale_h缩放
        query_points_scaled_x = (query_points_squeeze[:, 0] * scale_w).round().long()
        query_points_scaled_y = (query_points_squeeze[:, 1] * scale_h).round().long()
        # 合并坐标并保护（防止越界）
        query_points_scaled = torch.stack([query_points_scaled_x, query_points_scaled_y], dim=1)
        query_points_scaled = query_points_scaled.cpu().numpy()
        # 坐标越界保护
        query_points_scaled[:, 0] = np.clip(query_points_scaled[:, 0], 0, conf_w - 1)
        query_points_scaled[:, 1] = np.clip(query_points_scaled[:, 1], 0, conf_h - 1)

        pred_conf = conf[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]
        pred_point_3d = points_3d[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]

        # heuristic to remove low confidence points
        # should I export this as an input parameter?
        valid_mask = pred_conf > 1.2
        if valid_mask.sum() > 512:
            query_points = query_points[:, valid_mask]  # Make sure shape is compatible
            pred_conf = pred_conf[valid_mask]
            pred_point_3d = pred_point_3d[valid_mask]
            pred_color = pred_color[valid_mask]
    else:
        pred_conf = None
        pred_point_3d = None

    reorder_index = calculate_index_mappings(query_index, frame_num, device=device)

    images_feed, fmaps_feed = switch_tensor_order([images, fmaps_for_tracker], reorder_index, dim=0)
    images_feed = images_feed[None]  # add batch dimension
    fmaps_feed = fmaps_feed[None]  # add batch dimension

    all_points_num = images_feed.shape[1] * query_points.shape[1]

    # Don't need to be scared, this is just chunking to make GPU happy
    if all_points_num > max_points_num:
        num_splits = (all_points_num + max_points_num - 1) // max_points_num
        query_points = torch.chunk(query_points, num_splits, dim=1)
    else:
        query_points = [query_points]

    pred_track, pred_vis, _ = predict_tracks_in_chunks(
        tracker, images_feed, query_points, fmaps_feed, fine_tracking=fine_tracking
    )

    pred_track, pred_vis = switch_tensor_order([pred_track, pred_vis], reorder_index, dim=1)

    pred_track = pred_track.squeeze(0).float().cpu().numpy()
    pred_vis = pred_vis.squeeze(0).float().cpu().numpy()

    return pred_track, pred_vis, pred_conf, pred_point_3d, pred_color


# ===================== 修复：返回新增轨迹对应的【真实来源帧】 =====================
def _augment_non_visible_frames(
    pred_tracks: list,
    pred_vis_scores: list,
    pred_confs: list,
    pred_points_3d: list,
    pred_colors: list,
    images: torch.Tensor,
    conf,
    points_3d,
    fmaps_for_tracker,
    keypoint_extractors,
    tracker,
    max_points_num: int,
    fine_tracking: bool,
    *,
    min_vis: int = 500,
    non_vis_thresh: float = 0.1,
    device: torch.device = None,
):
    """
    🔥 核心修复：新增返回值 added_origins → 【每一条新增轨迹的真实来源query index】
    顺序与pred_tracks新增的轨迹完全对应
    """
    last_query = -1
    final_trial = False
    cur_extractors = keypoint_extractors
    added_origins = []  # ✅ 记录：新增轨迹 → 对应来源帧

    while True:
        vis_array = np.concatenate(pred_vis_scores, axis=1)
        sufficient_vis_count = (vis_array > non_vis_thresh).sum(axis=-1)
        non_vis_frames = np.where(sufficient_vis_count < min_vis)[0].tolist()
        if len(non_vis_frames) == 0:
            break

        print("Processing non visible frames:", non_vis_frames)
        if non_vis_frames[0] == last_query:
            final_trial = True
            cur_extractors = initialize_feature_extractors(2048, extractor_method="sp+sift+aliked", device=device)
            query_frame_list = non_vis_frames
        else:
            query_frame_list = [non_vis_frames[0]]

        last_query = non_vis_frames[0]
        # 遍历补轨迹的查询帧 → 记录来源
        for query_index in query_frame_list:
            new_track, new_vis, new_conf, new_point_3d, new_color = _forward_on_query(
                query_index, images, conf, points_3d, fmaps_for_tracker,
                cur_extractors, tracker, max_points_num, fine_tracking, device
            )
            # 添加轨迹
            pred_tracks.append(new_track)
            pred_vis_scores.append(new_vis)
            pred_confs.append(new_conf)
            pred_points_3d.append(new_point_3d)
            pred_colors.append(new_color)
            # ✅ 关键：记录【真实来源帧】（生成这条轨迹用的query_index）
            added_origins.append(query_index)

        if final_trial:
            break

    # ========== 新增返回：added_origins（真实来源，无-1） ==========
    return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors, added_origins


# ===================== generate_uniform_queries 完全不变 =====================
def generate_uniform_queries(frame_num, query_num, stride=None):
    if frame_num <= 0 or query_num <= 0:
        return []
    uniform_indices = np.linspace(0, frame_num - 1, num=query_num, endpoint=True)
    uniform_indices = np.round(uniform_indices).astype(int)
    uniform_indices = np.unique(uniform_indices).tolist()
    if len(uniform_indices) < query_num:
        all_frames = set(range(frame_num))
        missing_frames = sorted(list(all_frames - set(uniform_indices)))
        uniform_indices.extend(missing_frames[:query_num - len(uniform_indices)])
    return sorted(uniform_indices)[:query_num]

import numpy as np

def generate_uniform_queries(frame_num, query_num, stride=None):
    """
    真正的等间隔均匀选择查询帧，覆盖整个序列（从第0帧到最后一帧）
    Args:
        frame_num: 总帧数
        query_num: 需要选择的查询帧数
        stride: 废弃参数（保持兼容）
    Returns:
        排序后的均匀采样索引列表
    """
    if frame_num <= 0:
        return []
    if query_num <= 0:
        return []
    
    # 核心：用linspace生成均匀间隔的索引（覆盖0到frame_num-1）
    # endpoint=True：确保包含最后一帧，保证全局均匀
    uniform_indices = np.linspace(0, frame_num - 1, num=query_num, endpoint=True)
    # 四舍五入到最近的整数，并去重（避免帧数量少、查询数多导致重复）
    uniform_indices = np.round(uniform_indices).astype(int)
    uniform_indices = np.unique(uniform_indices).tolist()
    
    # 特殊情况：如果去重后数量不足，补充缺失的帧（优先补充间隔大的位置）
    if len(uniform_indices) < query_num:
        # 找出未被采样的帧
        all_frames = set(range(frame_num))
        missing_frames = sorted(list(all_frames - set(uniform_indices)))
        # 补充缺失的帧直到满足数量
        uniform_indices.extend(missing_frames[:query_num - len(uniform_indices)])
    
    # 排序并截断到目标数量（确保顺序）
    uniform_indices = sorted(uniform_indices)[:query_num]
    
    return uniform_indices