"""Differentiable validation rollout helpers."""

import math

import torch
from torch.func import functional_call
from torch.nn import functional as F

from worker_context_features import extract_worker_context_features
from .planner_utils import compute_global_guidance_meta_loss
from .tensor_utils import (
    build_command_velocity,
    build_yaw_frame,
    compute_action_energy_loss_per_step,
    compute_action_smoothness_losses_per_step,
    compute_arrival_reward,
    compute_goal_progress_preference_loss,
    compute_heading_reference,
    compute_stuck_loss,
    compute_velocity_heading_command,
    compute_velocity_tracking_loss,
    decode_worker_action,
    extract_depth_geometry_features,
    extract_progress_features,
    safe_l2_norm,
    select_min_clearance_obstacle_sample,
    sanitize_tensor,
)


def unrolled_meta_rollout(
    env,
    worknet,
    fast_params,
    args,
    B,
    device,
    potential_map_cache,
    global_planner,
    command_speed,
    iter_idx=0,
):
    """
    Validation rollout with virtually-updated worker params (via functional_call).
    Computes and returns meta_loss plus its actual weighted optimization terms.
    Progress, smoothness, and normalized control effort are included in the
    task-performance objective used to train LGN.
    LGN is NOT needed here; this meta loss is task-performance based.
    Reuses the same maze layout for consistent LGN signal.
    """
    # 保持同一张迷宫布局, 仅重置无人机状态用于验证rollout
    env.reset_drone_only()

    p_list, p_next_list, v_list, a_list, vec_list = [], [], [], [], []
    target_v_list = []
    dist_obj_list = []
    act_buf = [env.act.detach()] * 2
    h_val = None
    yaw_rate_max_value = math.radians(float(args.yaw_rate_max_deg))

    for t in range(args.lgn_timesteps):
        ctl_dt = 1.0 / 15.0
        depth, flow = env.render(ctl_dt)
        depth = sanitize_tensor(depth, nan=24.0, posinf=24.0, neginf=0.3)

        p_list.append(env.p)
        v_list.append(env.v)
        a_list.append(env.a)
        vec_curr_samples = env.find_vec_to_nearest_pt()
        vec_curr, dist_obj_curr = select_min_clearance_obstacle_sample(
            vec_curr_samples, env.margin, sample_dim=0,
        )
        vec_list.append(vec_curr)
        dist_obj_list.append(dist_obj_curr)

        target_v_raw = env.p_target - env.p.detach()
        target_v, _ = build_command_velocity(
            env.p.detach(), env.p_target, command_speed, args.goal_slowdown_time
        )
        target_v_list.append(target_v)

        R = build_yaw_frame(env.R) if args.attitude_model == 'v2' else env.R
        state_list = [torch.squeeze(target_v[:, None] @ R, 1), env.R[:, 2], env.margin[:, None]]
        if args.attitude_model == 'v2':
            heading_ref_world, heading_ref_local_xy, _ = compute_heading_reference(env, R)
            yaw_rate_norm = getattr(env, "yaw_rate", torch.zeros((B, 1), device=device)) / float(yaw_rate_max_value)
            state_list.extend([heading_ref_local_xy, yaw_rate_norm])
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom:
            state_list.insert(0, local_v)

        state_t = sanitize_tensor(
            torch.cat(state_list, -1),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        ).clamp(-10.0, 10.0)

        x_pooled = F.max_pool2d((3 / depth.clamp(0.3, 24) - 0.6)[:, None], 4, 4)
        x_pooled = sanitize_tensor(x_pooled, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

        geom_feat = extract_depth_geometry_features(depth)
        geom_feat = sanitize_tensor(
            geom_feat,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        ).clamp(-10.0, 10.0)

        progress_feat_base_raw = extract_progress_features(
            p_history_list=p_list,
            v_history_list=v_list,
            dist_obj_history_list=dist_obj_list,
            p_target=env.p_target,
            window=8,
        )
        context_feat_raw = extract_worker_context_features(
            p_history_list=p_list,
            p_target=env.p_target,
            R_current=R,
        )
        progress_feat = torch.cat([progress_feat_base_raw, context_feat_raw], dim=-1)
        progress_feat = sanitize_tensor(
            progress_feat,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        ).clamp(-10.0, 10.0)

        # Worker forward with virtually-updated params
        worker_input = torch.cat([state_t, geom_feat, progress_feat], dim=-1)
        act_out, _, h_val = functional_call(worknet, fast_params, (x_pooled, worker_input, h_val))
        act_out = sanitize_tensor(act_out, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

        a_pred, v_pred, yaw_rate_cmd = decode_worker_action(act_out, R, yaw_rate_max_value)
        real_act = (
            (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None]
            + env.g_std
        )
        real_act = sanitize_tensor(real_act, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        act_buf.append(real_act)

        if args.attitude_model == 'v2':
            heading_v_ref = env.v

            heading_ref_world, heading_ref_local_xy, yaw_error_vel, yaw_rate_rule, heading_speed_xy = \
                compute_velocity_heading_command(
                    R_yaw=R,
                    v_ref_world=heading_v_ref,
                    yaw_rate_max_value=yaw_rate_max_value,
                    yaw_kp=args.heading_yaw_kp,
                    min_speed=args.heading_min_speed,
                )

            if yaw_rate_cmd is None:
                yaw_rate_residual = torch.zeros((B, 1), device=device, dtype=real_act.dtype)
            else:
                yaw_rate_residual = yaw_rate_cmd

            yaw_rate_cmd_final = yaw_rate_rule + yaw_rate_residual
            yaw_rate_cmd_final = torch.clamp(
                yaw_rate_cmd_final,
                -float(yaw_rate_max_value),
                float(yaw_rate_max_value),
            )

            env.run(
                real_act,
                ctl_dt,
                heading_ref=heading_ref_world,
                yaw_rate_cmd=yaw_rate_cmd_final,
                yaw_rate_max=yaw_rate_max_value,
            )
        else:
            env.run(real_act, ctl_dt, target_v_raw)

        p_next_list.append(env.p)

        # Keep full horizon so in-goal staying can continuously accumulate arrival reward.

        # 周期性截断以限制显存
        if args.detach_interval > 0 and (t + 1) % args.detach_interval == 0:
            if h_val is not None:
                h_val = h_val.detach()

    # --- 计算 Meta Loss ---
    p_val = torch.stack(p_list)
    p_next_val = torch.stack(p_next_list)
    a_val = torch.stack(a_list)
    v_val = torch.stack(v_list)  # [T, B, 3]
    target_v_val = torch.stack(target_v_list)
    act_val = torch.stack(act_buf)
    vec_val = torch.stack(vec_list)
    dist_val = sanitize_tensor(torch.stack(dist_obj_list), nan=0.0, posinf=10.0, neginf=-10.0)

    collision_depth_val = F.relu(-dist_val)
    loss_stuck_val, loss_collision_duration_val, stuck_ratio = compute_stuck_loss(
        p_val, collision_depth_val,
        stuck_window=args.stuck_window,
        displacement_threshold=args.stuck_displacement_threshold,
    )

    # Final distance is retained as an evaluation metric, but it is deliberately
    # excluded from the outer objective that updates LGN. Compute it detached so
    # returning the metric does not keep the validation rollout graph alive.
    with torch.no_grad():
        m_pos = safe_l2_norm(p_val[-1] - env.p_target, dim=-1).mean()
    m_arrival_reward, m_arrival_hit_rate, m_arrival_best_dist = compute_arrival_reward(
        p_val,
        env.p_target,
        radius=args.meta_arrival_reward_radius,
        temperature=args.arrival_soft_temperature,
        time_step=ctl_dt,
    )
    with torch.no_grad():
        v_to_pt = torch.ones_like(dist_val)
        if dist_val.shape[0] > 1:
            v_to_pt[1:] = (-torch.diff(dist_val, 1, 0) * 135.0).clamp_min(1.0)
    m_coll = (F.softplus(dist_val.mul(-32.0)) * v_to_pt).mean()
    m_ctrl = safe_l2_norm(act_val, dim=-1).sum()
    progress_step_scale = max(float(env.max_speed) * float(ctl_dt), 1e-3)
    progress_loss_val = compute_goal_progress_preference_loss(
        p_val,
        p_next_val,
        env.p_target,
        step_scale=progress_step_scale,
    )
    smoothness_loss_val, jerk_loss_val, snap_loss_val = \
        compute_action_smoothness_losses_per_step(
            act_val,
            env.g_std,
            control_hz=1.0 / float(ctl_dt),
            jerk_weight=args.meta_smooth_jerk_weight,
            snap_weight=args.meta_smooth_snap_weight,
        )
    energy_loss_val = compute_action_energy_loss_per_step(
        act_val[2:],
        action_scale=args.energy_action_scale,
    )
    m_progress = progress_loss_val.mean()
    m_jerk = jerk_loss_val.mean()
    m_snap = snap_loss_val.mean()
    m_energy = energy_loss_val.mean()
    # Match the main branch: guard only the valid altitude range, with no
    # preferred in-range altitude.
    z_val = p_val[:, :, 2]
    m_height_bounds = (
        F.softplus((z_val - 5.0) * 20.0)
        + F.softplus((0.0 - z_val) * 20.0)
    ).mean()
    m_velocity_track = compute_velocity_tracking_loss(
        v_val,
        target_v_val,
        window=args.velocity_track_window,
    )
    m_stuck = loss_stuck_val.mean()

    if bool(getattr(args, "guidance_enabled", True)):
        m_guidance, _ = compute_global_guidance_meta_loss(
            env, p_val, v_val, env.p_target, vec_val, dist_val,
            config=args,
            potential_map_cache=potential_map_cache,
            planner=global_planner,
            a_history=a_val,
            sample_count=args.guide_sample_count,
            strategy=args.guide_sample_strategy,
            max_speed=float(env.max_speed),
            max_accel=args.guide_max_accel,
            max_decel=args.guide_max_decel,
            dir_weight=args.guide_dir_weight,
            speed_weight=args.guide_speed_weight,
            lateral_weight=args.guide_lateral_weight,
            escape_weight=args.guide_escape_weight,
            collision_threshold=args.guide_collision_threshold,
            accel_weight=args.guide_accel_weight,
            speed_diff_weight=args.guide_speed_diff_weight,
            recovery_speed_weight=args.guide_recovery_speed_weight,
        )
    else:
        m_guidance = m_coll.new_zeros(())

    # Keep the exact weighted terms that form the optimization objective so
    # diagnostics can probe each term's gradient to the virtual Worker.
    meta_terms = {
        "collision": m_coll,
        "height_bounds": m_height_bounds,
        "guidance_weighted": args.meta_guidance_weight * m_guidance,
        "smooth_jerk_weighted": args.meta_smooth_jerk_weight * m_jerk,
        "smooth_snap_weighted": args.meta_smooth_snap_weight * m_snap,
        "progress_weighted": args.meta_progress_weight * m_progress,
        "energy_weighted": args.meta_energy_weight * m_energy,
        "velocity_track_weighted": args.meta_velocity_track_weight * m_velocity_track,
        "stuck_weighted": args.stuck_loss_weight * m_stuck,
        "first_arrival_log_time_reward_weighted": -args.meta_arrival_reward_weight * m_arrival_reward,
    }
    meta_val = sum(meta_terms.values())
    return (
        meta_val,
        m_pos,
        m_coll,
        m_ctrl,
        m_arrival_reward,
        m_arrival_hit_rate,
        m_arrival_best_dist,
        meta_terms,
    )
