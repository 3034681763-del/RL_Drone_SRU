"""Tensor, feature extraction, and loss helpers."""

from collections import defaultdict
import math

import torch
from torch.nn import functional as F

from turn_loss_utils import (
    compute_direction_stability_loss_3d,
    compute_speed_stability_loss,
)


def safe_normalize(x, dim=-1, eps=1e-6):
    return F.normalize(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), dim=dim, eps=eps)


def safe_l2_norm(x, dim=-1, keepdim=False, eps=1e-6):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sqrt((x * x).sum(dim=dim, keepdim=keepdim) + eps)


def sample_command_speed(
    batch_size,
    min_speed,
    max_speed,
    speed_mtp,
    device,
    dtype=torch.float32,
    n_drones_per_group=1,
    randomize=True,
    fixed_speed=None,
):
    """Sample one commanded speed per rollout, matching the official grouping.

    In multi-drone mode every consecutive group receives the same sampled
    command.  ``speed_mtp`` scales random and fixed commands consistently.
    """
    batch_size = int(batch_size)
    group_size = int(n_drones_per_group)
    min_speed = float(min_speed)
    max_speed = float(max_speed)
    speed_mtp = float(speed_mtp)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if group_size <= 0:
        raise ValueError("n_drones_per_group must be positive")
    if not math.isfinite(min_speed) or min_speed < 0.0:
        raise ValueError("min_speed must be finite and non-negative")
    if not math.isfinite(max_speed) or max_speed < min_speed:
        raise ValueError("max_speed must be finite and >= min_speed")
    if not math.isfinite(speed_mtp) or speed_mtp <= 0.0:
        raise ValueError("speed_mtp must be finite and positive")

    group_count = (batch_size + group_size - 1) // group_size
    if randomize:
        group_speed = min_speed + (max_speed - min_speed) * torch.rand(
            (group_count, 1), device=device, dtype=dtype
        )
    else:
        selected = (min_speed + max_speed) * 0.5 if fixed_speed is None else float(fixed_speed)
        if not math.isfinite(selected) or selected < 0.0:
            raise ValueError("fixed_speed must be finite and non-negative")
        group_speed = torch.full(
            (group_count, 1), selected, device=device, dtype=dtype
        )
    return group_speed.repeat_interleave(group_size, dim=0)[:batch_size] * speed_mtp


def build_command_velocity(position, target, command_speed, slowdown_time=1.0):
    """Build a goal-directed command velocity with linear near-goal slowdown."""
    slowdown_time = float(slowdown_time)
    if slowdown_time <= 0.0 or not math.isfinite(slowdown_time):
        raise ValueError("slowdown_time must be finite and positive")
    if position.shape != target.shape or position.shape[-1] != 3:
        raise ValueError("position and target must have matching [..., 3] shapes")
    if command_speed.shape != position.shape[:-1] + (1,):
        raise ValueError(
            f"command_speed must have shape {position.shape[:-1] + (1,)}, "
            f"got {tuple(command_speed.shape)}"
        )

    goal_vector = target - position
    distance = safe_l2_norm(goal_vector, dim=-1, keepdim=True)
    goal_direction = goal_vector / distance.clamp_min(1e-6)
    effective_speed = torch.minimum(
        command_speed.clamp_min(0.0),
        distance / slowdown_time,
    )
    return goal_direction * effective_speed, effective_speed


def compute_velocity_tracking_loss(v_history, target_v_history, window=30):
    """Official-style Smooth-L1 loss on window-averaged velocity error.

    The normal branch preserves the upstream 30-step averaging and alignment.
    Short smoke-test rollouts use their full horizon so the loss stays defined.
    """
    if v_history.shape != target_v_history.shape or v_history.dim() != 3:
        raise ValueError("velocity histories must have matching [T, B, 3] shapes")
    if v_history.shape[0] <= 0:
        raise ValueError("velocity histories must contain at least one time step")
    window = int(window)
    if window <= 0:
        raise ValueError("window must be positive")
    if v_history.shape[0] > window:
        cumulative = torch.cumsum(v_history, dim=0)
        averaged = (cumulative[window:] - cumulative[:-window]) / float(window)
        target_aligned = target_v_history[1:1 - window]
    else:
        averaged = v_history.mean(dim=0, keepdim=True)
        target_aligned = target_v_history[:1]
    delta_v = torch.linalg.vector_norm(
        torch.nan_to_num(averaged - target_aligned, nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
    )
    return F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))


def select_min_clearance_obstacle_sample(vec_samples, margin, sample_dim=0):
    """Select the look-ahead sample with the smallest obstacle clearance.

    Distances must be reduced before vectors: averaging vectors from different
    samples can cancel opposing directions and hide the dangerous sample.
    """
    if vec_samples.ndim < 2:
        raise ValueError(f"vec_samples must include a sample axis, got shape={tuple(vec_samples.shape)}")

    sample_dim = int(sample_dim)
    if sample_dim < 0:
        sample_dim += vec_samples.ndim
    if sample_dim < 0 or sample_dim >= vec_samples.ndim - 1:
        raise ValueError(
            f"sample_dim must index a non-coordinate axis, got sample_dim={sample_dim}, "
            f"shape={tuple(vec_samples.shape)}"
        )

    clearance_samples = safe_l2_norm(vec_samples, dim=-1) - margin
    min_clearance, min_idx = clearance_samples.min(dim=sample_dim)
    gather_idx = min_idx.unsqueeze(sample_dim).unsqueeze(-1)
    gather_shape = list(vec_samples.shape)
    gather_shape[sample_dim] = 1
    gather_idx = gather_idx.expand(*gather_shape)
    selected_vec = torch.gather(vec_samples, sample_dim, gather_idx).squeeze(sample_dim)
    return selected_vec, min_clearance


def compute_arrival_reward(
    p_history,
    p_target,
    radius=0.5,
    temperature=0.05,
    time_step=1.0 / 15.0,
):
    """Differentiable reward for an early *first* arrival at the goal.

    A sigmoid turns each distance into a soft hit probability ``q_t``.  The
    probability that the first hit occurs at step ``t`` is

        f_t = q_t * product_{k<t}(1 - q_k).

    Products are evaluated as an exclusive cumulative sum in log space for
    numerical stability.  First-hit probability is weighted by a normalized
    logarithmic time score, so an earlier first arrival receives more reward
    and never arriving receives zero reward.  Remaining inside the goal does
    not repeatedly receive the full arrival reward.

    The returned hit rate and best distance deliberately remain hard,
    no-gradient evaluation metrics.
    """
    radius = float(radius)
    temperature = float(temperature)
    time_step = float(time_step)
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if time_step <= 0.0:
        raise ValueError("time_step must be positive")
    if p_history.shape[0] <= 0:
        raise ValueError("p_history must contain at least one time step")

    target = p_target.unsqueeze(0) if p_history.dim() == 3 else p_target
    dist_to_goal = safe_l2_norm(p_history - target, dim=-1)

    hit_logit = (radius - dist_to_goal) / temperature
    log_hit = F.logsigmoid(hit_logit)
    log_not_hit = F.logsigmoid(-hit_logit)

    # survival_before[t] = product_{k<t}(1 - q_k).  Prepending log(1)=0
    # makes the cumulative sum exclusive without detaching the trajectory.
    log_not_hit_cumulative = torch.cumsum(log_not_hit, dim=0)
    log_survival_before = torch.cat(
        [torch.zeros_like(log_not_hit_cumulative[:1]), log_not_hit_cumulative[:-1]],
        dim=0,
    )
    first_hit_probability = torch.exp(log_survival_before + log_hit)

    horizon_steps = dist_to_goal.shape[0]
    time_seconds = torch.arange(
        horizon_steps,
        device=dist_to_goal.device,
        dtype=dist_to_goal.dtype,
    ) * time_step
    missed_time_seconds = dist_to_goal.new_tensor(float(horizon_steps) * time_step)
    log_time_denominator = torch.log1p(missed_time_seconds)
    log_time_score = 1.0 - torch.log1p(time_seconds) / log_time_denominator
    log_time_score = log_time_score.reshape(
        (horizon_steps,) + (1,) * (dist_to_goal.dim() - 1)
    )

    reward_per_agent = (first_hit_probability * log_time_score).sum(dim=0)
    reward = reward_per_agent.mean()
    with torch.no_grad():
        hit_rate = (dist_to_goal <= radius).any(dim=0).float().mean()
        best_dist = dist_to_goal.min(dim=0).values.mean()
    return reward, hit_rate, best_dist


def build_yaw_frame(R):
    fwd = R[:, :, 0]
    zeros = torch.zeros_like(fwd)
    up = zeros.clone()
    up[:, 2] = 1.0
    fwd_h_raw = torch.stack([fwd[:, 0], fwd[:, 1], torch.zeros_like(fwd[:, 2])], dim=-1)
    fwd_h_norm = safe_l2_norm(fwd_h_raw, dim=-1, keepdim=True)
    fallback = zeros.clone()
    fallback[:, 0] = 1.0
    fwd_h = torch.where(fwd_h_norm > 1e-6, fwd_h_raw / fwd_h_norm.clamp_min(1e-6), fallback)
    left = safe_normalize(torch.cross(up, fwd_h, dim=-1), dim=-1)
    return torch.stack([fwd_h, left, up], -1)


def compute_heading_reference(env, R_yaw):
    target_vec = env.p_target - env.p.detach()
    zeros = torch.zeros_like(target_vec[:, 2])
    heading_ref_world = torch.stack([target_vec[:, 0], target_vec[:, 1], zeros], dim=-1)
    heading_norm = safe_l2_norm(heading_ref_world, dim=-1, keepdim=True)
    fallback = R_yaw[:, :, 0]
    heading_ref_world = torch.where(
        heading_norm > 1e-6,
        heading_ref_world / heading_norm.clamp_min(1e-6),
        fallback,
    )
    heading_ref_local = torch.squeeze(heading_ref_world[:, None] @ R_yaw, 1)
    yaw_error = torch.atan2(heading_ref_local[:, 1], heading_ref_local[:, 0]).unsqueeze(-1)
    return heading_ref_world, heading_ref_local[:, :2], yaw_error


def compute_velocity_heading_command(
    R_yaw,
    v_ref_world,
    yaw_rate_max_value,
    yaw_kp=4.0,
    min_speed=0.25,
):
    """
    根据期望速度方向计算机头参考方向和 yaw_rate_cmd。

    逻辑：
    1. 只使用水平面速度方向，不让 z 方向影响 yaw；
    2. 机头参考方向始终来自速度方向（不再做低速保持当前机头）；
    3. yaw_rate_cmd = yaw_kp * yaw_error，并限制在 [-yaw_rate_max, yaw_rate_max]；
    4. 即使 v_ref_world 反向，也不会让机头瞬间跳 180°，而是通过 yaw_rate_max 平滑转过去。
    """
    _ = min_speed  # kept for call-site compatibility; no low-speed heading hold is applied.

    v_xy = torch.stack([
        v_ref_world[:, 0],
        v_ref_world[:, 1],
        torch.zeros_like(v_ref_world[:, 2]),
    ], dim=-1)
    speed_xy = safe_l2_norm(v_xy, dim=-1, keepdim=True)
    heading_ref_world = safe_normalize(v_xy, dim=-1)

    heading_ref_local = torch.squeeze(heading_ref_world[:, None] @ R_yaw, 1)
    yaw_error = torch.atan2(
        heading_ref_local[:, 1],
        heading_ref_local[:, 0],
    ).unsqueeze(-1)

    yaw_rate_cmd = torch.clamp(
        float(yaw_kp) * yaw_error,
        -float(yaw_rate_max_value),
        float(yaw_rate_max_value),
    )

    return heading_ref_world, heading_ref_local[:, :2], yaw_error, yaw_rate_cmd, speed_xy


def decode_worker_action(act, R_yaw, yaw_rate_max_value):
    """Decode official interleaved acceleration/velocity outputs.

    Layout is ``[ax, vx, ay, vy, az, vz, yaw]`` for v2 and the same first six
    channels without yaw for legacy attitude dynamics.
    """
    if act.dim() != 2 or act.shape[-1] not in (6, 7):
        raise ValueError(f"Worker action must have 6 or 7 channels, got {tuple(act.shape)}")
    accel_body, velocity_body = act[:, :6].reshape(act.shape[0], 3, 2).unbind(-1)
    a_pred = torch.squeeze(R_yaw @ accel_body.unsqueeze(-1), -1)
    v_pred = torch.squeeze(R_yaw @ velocity_body.unsqueeze(-1), -1)
    yaw_rate_cmd = None
    if act.shape[-1] == 7:
        yaw_rate_cmd = torch.tanh(act[:, 6:7]) * float(yaw_rate_max_value)
    return a_pred, v_pred, yaw_rate_cmd


def sanitize_tensor(x, nan=0.0, posinf=1e3, neginf=-1e3):
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def extract_depth_geometry_features(depth, near_threshold=1.5):
    """
    从原始深度图提取显式几何/风险统计特征。

    输入:
        depth: [B, H, W]
    输出:
        geom_feat: [B, 19]
    """
    d = depth.clamp(0.3, 24.0)
    B, H, W = d.shape

    w1 = max(1, W // 3)
    w2 = max(w1 + 1, (2 * W) // 3)
    w2 = min(w2, W - 1) if W > 2 else w2

    h1 = max(1, H // 3)
    h2 = max(h1 + 1, (2 * H) // 3)
    h2 = min(h2, H - 1) if H > 2 else h2

    left = d[:, :, :w1]
    center = d[:, :, w1:w2]
    right = d[:, :, w2:]

    upper = d[:, :h1, :]
    middle = d[:, h1:h2, :]
    lower = d[:, h2:, :]

    def _mean_min_ratio(region):
        flat = region.reshape(B, -1)
        mean_v = flat.mean(dim=-1)
        min_v = flat.min(dim=-1).values
        near_ratio = (flat < near_threshold).float().mean(dim=-1)
        return mean_v, min_v, near_ratio

    def _mean_ratio(region):
        flat = region.reshape(B, -1)
        mean_v = flat.mean(dim=-1)
        near_ratio = (flat < near_threshold).float().mean(dim=-1)
        return mean_v, near_ratio

    l_mean, l_min, l_ratio = _mean_min_ratio(left)
    c_mean, c_min, c_ratio = _mean_min_ratio(center)
    r_mean, r_min, r_ratio = _mean_min_ratio(right)

    u_mean, u_ratio = _mean_ratio(upper)
    m_mean, m_ratio = _mean_ratio(middle)
    lo_mean, lo_ratio = _mean_ratio(lower)

    flat_all = d.reshape(B, -1)
    g_mean = flat_all.mean(dim=-1)
    g_std = flat_all.std(dim=-1, unbiased=False)
    lr_diff = l_mean - r_mean
    center_vs_side = c_mean - 0.5 * (l_mean + r_mean)

    geom_feat = torch.stack([
        l_mean, l_min, l_ratio,
        c_mean, c_min, c_ratio,
        r_mean, r_min, r_ratio,
        u_mean, u_ratio,
        m_mean, m_ratio,
        lo_mean, lo_ratio,
        g_mean, g_std, lr_diff, center_vs_side,
    ], dim=-1)
    return sanitize_tensor(geom_feat, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)


def extract_progress_features(p_history_list, v_history_list, dist_obj_history_list, p_target, window=8):
    """
    构造近期进展/卡住摘要特征。

    输出维度: [B, 8]
    - progress_to_goal
    - disp_k
    - speed_mean_k
    - collision_depth_mean_k
    - stuck_score
    - progress_efficiency
    - tortuosity
    - heading_align_improvement
    """
    if len(p_history_list) == 0:
        B = p_target.shape[0]
        return torch.zeros((B, 8), device=p_target.device, dtype=p_target.dtype)

    p_now = p_history_list[-1]
    device = p_now.device
    dtype = p_now.dtype

    k = max(1, min(int(window), len(p_history_list)))
    p_prev = p_history_list[-k]

    dist_now = safe_l2_norm(p_target - p_now, dim=-1)
    dist_prev = safe_l2_norm(p_target - p_prev, dim=-1)
    progress_to_goal = dist_prev - dist_now

    disp_k = safe_l2_norm(p_now - p_prev, dim=-1)

    v_tail = torch.stack(v_history_list[-k:], dim=0)
    speed_mean_k = safe_l2_norm(v_tail, dim=-1).mean(dim=0)

    if len(dist_obj_history_list) > 0:
        dist_tail = torch.stack(dist_obj_history_list[-k:], dim=0)
        depth_tail = F.relu(-dist_tail)
        # dist_tail 可能是 [k, B] 或 [k, sub_div, B]，统一压缩到 [B]
        while depth_tail.dim() > 1:
            depth_tail = depth_tail.mean(dim=0)
        collision_depth_mean_k = depth_tail
    else:
        collision_depth_mean_k = torch.zeros_like(progress_to_goal)

    stuck_score = F.softplus((0.3 - disp_k) * 10.0)
    # 效率比: 跑了多少净位移是否真正转化为接近目标
    progress_efficiency = progress_to_goal / (disp_k + 1e-6)

    # 曲折度: 窗口路径长度 / 窗口净位移，越大表示越绕/打转
    p_tail = torch.stack(p_history_list[-k:], dim=0)  # [k, B, 3]
    if k > 1:
        path_len_k = safe_l2_norm(p_tail[1:] - p_tail[:-1], dim=-1).sum(dim=0)
    else:
        path_len_k = torch.zeros_like(disp_k)
    tortuosity = path_len_k / (disp_k + 1e-6)

    v_now = v_history_list[-1]
    target_dir_now = safe_normalize(p_target - p_now, dim=-1)
    v_dir_now = safe_normalize(v_now, dim=-1)
    heading_align_now = (v_dir_now * target_dir_now).sum(dim=-1)

    if len(v_history_list) >= k:
        v_prev = v_history_list[-k]
        target_dir_prev = safe_normalize(p_target - p_prev, dim=-1)
        v_dir_prev = safe_normalize(v_prev, dim=-1)
        heading_align_prev = (v_dir_prev * target_dir_prev).sum(dim=-1)
        heading_align_improvement = heading_align_now - heading_align_prev
    else:
        heading_align_improvement = torch.zeros_like(heading_align_now)

    progress_feat = torch.stack([
        progress_to_goal,
        disp_k,
        speed_mean_k,
        collision_depth_mean_k,
        stuck_score,
        progress_efficiency,
        tortuosity,
        heading_align_improvement,
    ], dim=-1)

    progress_feat = sanitize_tensor(progress_feat, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
    return progress_feat.to(device=device, dtype=dtype)


@torch.no_grad()
def sanitize_module_(module, clamp_value=10.0):
    for p in module.parameters():
        p.data = sanitize_tensor(p.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value).clamp(-clamp_value, clamp_value)


def rotation_matrix_to_rpy_deg(R):
    """Convert rotation matrix to roll-pitch-yaw in degrees (ZYX convention)."""
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]
    r10 = R[..., 1, 0]
    r00 = R[..., 0, 0]

    pitch = torch.asin(torch.clamp(-r20, -1.0, 1.0))
    roll = torch.atan2(r21, r22)
    yaw = torch.atan2(r10, r00)
    return torch.rad2deg(torch.stack([roll, pitch, yaw], dim=-1))


def merge_intervals(intervals, min_gap=1e-4):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + min_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


@torch.no_grad()
def get_collision_wall_patches(points_xyz, walls, drone_radius, segment_len=0.45, contact_eps=0.02):
    if points_xyz.numel() == 0 or walls.numel() == 0:
        return []

    wall_mask = (
        (walls[:, 2] >= 0.1)
        & (walls[:, 2] <= 1.9)
        & (walls[:, 5] > 0.5)
    )
    walls = walls[wall_mask]
    if walls.numel() == 0:
        return []

    centers = walls[:, :3]
    half = walls[:, 3:]
    wall_min = centers - half
    wall_max = centers + half
    axis_is_x = half[:, 0] <= half[:, 1]

    points_expanded = points_xyz.unsqueeze(1)
    nearest = torch.minimum(torch.maximum(points_expanded, wall_min.unsqueeze(0)), wall_max.unsqueeze(0))
    clearance = (nearest - points_expanded).norm(dim=-1) - float(drone_radius)
    contact_steps = torch.nonzero(clearance.min(dim=1).values <= contact_eps, as_tuple=False).flatten().tolist()

    wall_intervals = defaultdict(list)
    for step_idx in contact_steps:
        wall_idx = int(clearance[step_idx].argmin().item())
        if float(clearance[step_idx, wall_idx].item()) > contact_eps:
            continue

        point = points_xyz[step_idx]
        center = centers[wall_idx]
        wall_half = half[wall_idx]

        if bool(axis_is_x[wall_idx]):
            tangent_min = float(center[1] - wall_half[1])
            tangent_max = float(center[1] + wall_half[1])
            tangent_center = min(max(float(point[1]), tangent_min), tangent_max)
        else:
            tangent_min = float(center[0] - wall_half[0])
            tangent_max = float(center[0] + wall_half[0])
            tangent_center = min(max(float(point[0]), tangent_min), tangent_max)

        seg_half = min(0.5 * float(segment_len), 0.5 * (tangent_max - tangent_min))
        seg_start = max(tangent_min, tangent_center - seg_half)
        seg_end = min(tangent_max, tangent_center + seg_half)
        if seg_end - seg_start <= 1e-4:
            continue
        wall_intervals[wall_idx].append((seg_start, seg_end))

    patches = []
    for wall_idx, intervals in wall_intervals.items():
        center = centers[wall_idx]
        wall_half = half[wall_idx]
        for seg_start, seg_end in merge_intervals(intervals, min_gap=0.02):
            if bool(axis_is_x[wall_idx]):
                patches.append({
                    'xy': (float(center[0] - wall_half[0]), seg_start),
                    'width': float(2.0 * wall_half[0]),
                    'height': float(seg_end - seg_start),
                })
            else:
                patches.append({
                    'xy': (seg_start, float(center[1] - wall_half[1])),
                    'width': float(seg_end - seg_start),
                    'height': float(2.0 * wall_half[1]),
                })
    return patches


def compute_overlap_loss_per_step(p_history, sigma=0.5, time_window=10):
    """
    Step-wise 重叠损失计算
    返回: [Batch, Time] (注意: 调用处需要permute)
    """
    # Use squared-distance RBF directly (without sqrt/cdist) so 2nd-order gradients
    # stay well-behaved when trajectory points overlap exactly.
    p_history = p_history.permute(1, 0, 2)  # [B, T, 3]
    n_batch, n_points, _ = p_history.shape
    device = p_history.device
    dtype = p_history.dtype

    time_window = max(0, int(time_window))
    sigma = max(float(sigma), 1e-4)

    if n_points <= 1:
        return torch.zeros((n_batch, n_points), device=device, dtype=dtype)
    # Keep at least one valid long-range pair. Otherwise the mask becomes all-zero
    # and exploration term is permanently zero when time_window ~= rollout length.
    max_effective_window = max(0, n_points - 2)
    time_window = min(time_window, max_effective_window)

    # Pairwise squared distances: [B, T, T]
    pair_delta = p_history[:, :, None, :] - p_history[:, None, :, :]
    sq_dist = (pair_delta * pair_delta).sum(dim=-1)
    sq_dist = sanitize_tensor(sq_dist, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)
    inv_two_sigma2 = 0.5 / (sigma * sigma)
    overlap_energy = torch.exp(-sq_dist * inv_two_sigma2)

    indices = torch.arange(n_points, device=device)
    time_diff = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
    mask = (time_diff > time_window).to(dtype=dtype)  # [T, T]

    # Step-wise mean overlap energy with temporal exclusion mask.
    energy_sum = (overlap_energy * mask.unsqueeze(0)).sum(dim=2)  # [B, T]
    mask_sum = mask.sum(dim=1).unsqueeze(0).clamp_min(1.0)  # [1, T]

    return energy_sum / mask_sum


def compute_goal_progress_preference_loss(
    p_current,
    p_next,
    p_target,
    step_scale,
):
    """Return a bounded signed goal-progress preference for each action step.

    ``p_current[t]`` is the position before action ``t`` and ``p_next[t]`` is
    the position after that action. Moving closer to the goal produces a
    negative value (a reward under minimization), moving away produces a
    positive value, and no change produces zero. The tanh bound keeps this
    preference in ``[-1, 1]`` for stable LGN weighting and higher derivatives.
    """
    if p_current.shape != p_next.shape:
        raise ValueError(
            f"p_current and p_next must have identical shapes, got "
            f"{tuple(p_current.shape)} and {tuple(p_next.shape)}"
        )
    if p_current.shape[-1] != 3:
        raise ValueError(f"positions must end in xyz coordinates, got {tuple(p_current.shape)}")

    scale = float(step_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("step_scale must be finite and positive")

    target = p_target
    while target.dim() < p_current.dim():
        target = target.unsqueeze(0)
    dist_current = safe_l2_norm(p_current - target, dim=-1)
    dist_next = safe_l2_norm(p_next - target, dim=-1)
    return torch.tanh((dist_next - dist_current) / scale)


def compute_action_smoothness_losses_per_step(
    act_buffer,
    gravity_reference,
    control_hz=15.0,
    jerk_weight=0.001,
    snap_weight=0.0002,
):
    """Compute per-action jerk, snap, and their meta-consistent combination.

    ``act_buffer`` must contain two history actions followed by the ``T``
    actions being scored, so all returned tensors have shape ``[T, B]``.
    """
    if act_buffer.dim() != 3 or act_buffer.shape[-1] != 3:
        raise ValueError(f"act_buffer must have shape [T+2, B, 3], got {tuple(act_buffer.shape)}")
    if act_buffer.shape[0] < 3:
        raise ValueError("act_buffer must contain two history actions and at least one scored action")

    hz = float(control_hz)
    if not math.isfinite(hz) or hz <= 0.0:
        raise ValueError("control_hz must be finite and positive")

    # diff()[1:] aligns action t with its immediately preceding command; the
    # first diff is only between the two seed-history actions.
    jerk = act_buffer.diff(1, 0)[1:].mul(hz).pow(2).sum(dim=-1)
    thrust_direction = F.normalize(act_buffer - gravity_reference, dim=-1)
    snap = thrust_direction.diff(1, 0).diff(1, 0).mul(hz ** 2).pow(2).sum(dim=-1)
    smoothness = float(jerk_weight) * jerk + float(snap_weight) * snap
    return smoothness, jerk, snap


def compute_action_energy_loss_per_step(actions, action_scale=10.0):
    """Return normalized squared action magnitude as a control-effort proxy."""
    if actions.shape[-1] != 3:
        raise ValueError(f"actions must end in xyz acceleration, got {tuple(actions.shape)}")
    scale = float(action_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("action_scale must be finite and positive")
    return actions.div(scale).pow(2).sum(dim=-1)


def compute_turn_preference_loss(
    v_history,
    speed_threshold=0.2,
    speed_softness=0.01,
    soft_angle_deg=10.0,
):
    """
    三维速度方向稳定损失。

    相邻方向一致时损失为零，损失随夹角单调增加；小角度区域
    使用二次惩罚，并通过可微低速软掩码降低低速方向噪声的影响。

    输入:
        v_history: [T, B, 3]
    输出:
        loss_turn_seq: [T, B]
    """
    return compute_direction_stability_loss_3d(
        v_history=v_history,
        speed_threshold=speed_threshold,
        speed_softness=speed_softness,
        soft_angle_deg=soft_angle_deg,
    )


def compute_stuck_loss(p_history, collision_depth, stuck_window=15, displacement_threshold=0.3):
    """
    计算卡住惩罚损失

    检测两种卡住状态：
    1. 局部窗口内位移过小
    2. 持续碰撞状态
    """
    T, B, _ = p_history.shape
    device = p_history.device

    loss_stuck = torch.zeros((T, B), device=device)
    if T > stuck_window:
        for t in range(stuck_window, T):
            window_start = t - stuck_window
            displacement = safe_l2_norm(p_history[t] - p_history[window_start], dim=-1)  # [B]
            loss_stuck[t] = F.softplus((displacement_threshold - displacement) * 10.0)

    in_collision = (collision_depth > 0).float()  # [T, B]
    loss_collision_duration = torch.zeros_like(in_collision)

    collision_streak = torch.zeros((B,), device=device)
    for t in range(T):
        collision_streak = collision_streak * in_collision[t] + in_collision[t]
        loss_collision_duration[t] = collision_streak * in_collision[t]

    with torch.no_grad():
        stuck_mask = loss_stuck > 0.5
        stuck_ratio = stuck_mask.float().mean()

    return loss_stuck, loss_collision_duration, stuck_ratio
