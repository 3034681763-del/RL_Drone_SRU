from collections import defaultdict
import datetime
import json
import math
import os
from random import normalvariate
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from env_multi import Env
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse
from model import Model
from potential_map_utils import PotentialMapCache
from utils.map_utils import (
    _align_env_goal_planes_to_precomputed_map,
    _build_precomputed_map_type_indices,
)
from utils.visualization_utils import save_cached_viz_record, snapshot_env_for_viz

matplotlib.use('Agg', force=True)

try:
    import plotly.graph_objects as go
except Exception:
    go = None

###########参数配置##########

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_iters', type=int, default=500000)
parser.add_argument('--coef_v', type=float, default=1.0, help='smooth l1 of norm(v_set - v_real)')
parser.add_argument('--coef_speed', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_v_pred', type=float, default=2.0, help='mse loss for velocity estimation (no odom)')
parser.add_argument('--coef_collide', type=float, default=2.0, help='softplus loss for collision (large if close to obstacle, zero otherwise)')
parser.add_argument('--coef_obj_avoidance', type=float, default=1.5, help='quadratic clearance loss')
parser.add_argument('--coef_d_acc', type=float, default=0.01, help='control acceleration regularization')
parser.add_argument('--coef_d_jerk', type=float, default=0.001, help='control jerk regularizatinon')
parser.add_argument('--coef_d_snap', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_height_track', type=float, default=1.0,
                    help='track a smooth linear altitude reference from start z to goal z')
parser.add_argument('--coef_bias', type=float, default=0.0, help='legacy')
parser.add_argument('--attitude_model', type=str, default='v2', choices=['legacy', 'v2'],
                    help='legacy uses v_pred-projected heading; v2 uses explicit yaw-rate dynamics')
parser.add_argument('--yaw_rate_max_deg', type=float, default=150.0)
parser.add_argument('--coef_yaw_cmd', type=float, default=0.2)
parser.add_argument('--coef_yaw_smooth', type=float, default=0.01)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--grad_decay', type=float, default=0.4)
parser.add_argument('--speed_mtp', type=float, default=1.0)
parser.add_argument('--obstacle_count_scale', type=float, default=0.5,
                    help='global multiplier for obstacle counts')
parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
parser.add_argument('--timesteps', type=int, default=150)
parser.add_argument('--goal_radius', type=float, default=0.5,
                    help='Success goal proximity radius (meters)')
parser.add_argument('--maze_update_interval', type=int, default=50,
                    help='Switch precomputed map (or regenerate online map) every N iterations')
parser.add_argument('--precomputed_geometry_map_dir', type=str,
                    default='/home/robot/transformer/precomputed_geometry_density2x_tiled3x3',
                    help='mmgj_transformer precomputed geometry-map directory')
parser.add_argument('--num_precomputed_maps', type=int, default=0,
                    help='Maximum maps to load (<=0 means all); easy maps are used for training')
parser.add_argument('--use_precomputed_geometry_maps', dest='use_precomputed_geometry_maps',
                    action='store_true', help='Train on mmgj_transformer precomputed easy maps')
parser.add_argument('--no_precomputed_geometry_maps', dest='use_precomputed_geometry_maps',
                    action='store_false', help='Use the original online-generated maps')
parser.set_defaults(use_precomputed_geometry_maps=True)
parser.add_argument('--ball_obstacles', dest='use_ball_obstacles', action='store_true',
                    help='Keep spherical obstacles from the selected map')
parser.add_argument('--no_ball_obstacles', dest='use_ball_obstacles', action='store_false',
                    help='Remove spherical obstacles from rendering, collision checks and training')
parser.set_defaults(use_ball_obstacles=True)
parser.add_argument('--cam_angle', type=int, default=10)
parser.add_argument('--terminal_log_interval', type=int, default=500,
                    help='Update terminal progress text every N iterations')
parser.add_argument('--artifact_save_interval', type=int, default=1000,
                    help='Save mmgj-style trajectory HTML/video every N iterations (<=0 disables)')
parser.add_argument('--trajectory_save_interval', type=int, default=None,
                    help='Deprecated alias for --artifact_save_interval')
parser.add_argument('--single', default=False, action='store_true')
parser.add_argument('--gate', default=False, action='store_true')
parser.add_argument('--ground_voxels', default=False, action='store_true')
parser.add_argument('--scaffold', default=False, action='store_true')
parser.add_argument('--random_rotation', default=False, action='store_true')
parser.add_argument('--yaw_drift', default=False, action='store_true')
parser.add_argument('--no_odom', default=False, action='store_true')
parser.add_argument('--include_u_local_optimum', dest='include_u_local_optimum', action='store_true')
parser.add_argument('--no_include_u_local_optimum', dest='include_u_local_optimum', action='store_false')
parser.set_defaults(include_u_local_optimum=False)
parser.add_argument('--compact_two_zone_map', dest='compact_two_zone_map', action='store_true')
parser.add_argument('--no_compact_two_zone_map', dest='compact_two_zone_map', action='store_false')
parser.set_defaults(compact_two_zone_map=False)
parser.add_argument('--wall_physical_feedback', dest='wall_physical_feedback', action='store_true')
parser.add_argument('--no_wall_physical_feedback', dest='wall_physical_feedback', action='store_false')
parser.set_defaults(wall_physical_feedback=False)
parser.add_argument('--exp_name', type=str, default='main_cuda')
args = parser.parse_args()
if args.trajectory_save_interval is not None:
    args.artifact_save_interval = int(args.trajectory_save_interval)
# main_cuda loads geometry only; this compatibility flag is consumed by the
# shared mmgj visualization helper when deciding whether to draw potentials.
args.use_precomputed_potential_maps = False

current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_dir = os.path.join('..', 'checkpoints', f"main_cuda_{args.exp_name}_{current_time}")
video_dir = os.path.join(save_dir, 'videos')
os.makedirs(video_dir, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
print(args)
print(f"Training artifacts will be saved to: {save_dir}")
with open(os.path.join(save_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

device = torch.device('cuda')
yaw_rate_max = math.radians(float(args.yaw_rate_max_deg))
use_attitude_v2 = args.attitude_model == 'v2'

##########初始化仿真环境##########
env = Env(args.batch_size, 64, 48, args.grad_decay, device,
          fov_x_half_tan=args.fov_x_half_tan, single=args.single,
          gate=args.gate, ground_voxels=args.ground_voxels,
          scaffold=args.scaffold, speed_mtp=args.speed_mtp,
          random_rotation=args.random_rotation, cam_angle=args.cam_angle,
          obstacle_count_scale=args.obstacle_count_scale,
          include_u_local_optimum=args.include_u_local_optimum,
          compact_two_zone_map=args.compact_two_zone_map,
          wall_physical_feedback=args.wall_physical_feedback)

##########加载 mmgj_transformer 预计算地图##########
precomputed_map_cache = None
precomputed_easy_indices = []
current_precomputed_map_idx = -1
current_precomputed_map_file = ""
precomputed_map_update_count = 0
if args.use_precomputed_geometry_maps:
    precomputed_map_cache = PotentialMapCache(
        map_dir=args.precomputed_geometry_map_dir,
        num_maps=args.num_precomputed_maps,
    )
    if len(precomputed_map_cache) <= 0:
        raise RuntimeError(
            f"No precomputed maps found in {args.precomputed_geometry_map_dir}."
        )
    precomputed_type_indices = _build_precomputed_map_type_indices(precomputed_map_cache)
    precomputed_easy_indices = list(precomputed_type_indices.get('easy', []))
    if not precomputed_easy_indices:
        counts_msg = ", ".join(
            f"{map_type}={len(indices)}"
            for map_type, indices in sorted(precomputed_type_indices.items())
        )
        raise RuntimeError(
            "main_cuda mmgj-map training requires at least one easy precomputed map. "
            f"loaded_counts=({counts_msg}). If --num_precomputed_maps is set, "
            "increase it or use 0 to load all maps."
        )
    counts_msg = ", ".join(
        f"{map_type}={len(indices)}"
        for map_type, indices in sorted(precomputed_type_indices.items())
    )
    print(
        f"[PrecomputedMap] loaded={len(precomputed_map_cache)} "
        f"from {args.precomputed_geometry_map_dir}, easy-only training, "
        f"counts=({counts_msg})"
    )
if not args.use_ball_obstacles:
    print('[ObstacleFilter] spherical obstacles disabled; cylinders and voxels are kept')

##########初始化神经网络##########
base_state_dim = 7 if args.no_odom else 10
state_dim = base_state_dim + (3 if use_attitude_v2 else 0)
action_dim = 7 if use_attitude_v2 else 6
model = Model(state_dim, action_dim)
model = model.to(device)

##########使用预训练模型/继续训练原有的模型##########
if args.resume:
    state_dict = torch.load(args.resume, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    target_state = model.state_dict()
    compatible_state = {}
    resized_keys = []
    skipped_keys = []
    for key, value in state_dict.items():
        if key not in target_state:
            skipped_keys.append(key)
            continue
        target_value = target_state[key]
        if value.shape == target_value.shape:
            compatible_state[key] = value
            continue
        if value.dim() != target_value.dim():
            skipped_keys.append(key)
            continue
        expanded = torch.zeros_like(target_value)
        slices = tuple(slice(0, min(value.shape[d], target_value.shape[d])) for d in range(value.dim()))
        expanded[slices] = value[slices]
        compatible_state[key] = expanded
        resized_keys.append((key, tuple(value.shape), tuple(target_value.shape)))
    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, False)
    if missing_keys:
        print("missing_keys:", missing_keys)
    if unexpected_keys:
        print("unexpected_keys:", unexpected_keys)
    if resized_keys:
        print("resized_keys:", resized_keys)
    if skipped_keys:
        print("skipped_keys:", skipped_keys)

##########优化器##########
optim = AdamW(model.parameters(), args.lr)
##########学习率调度器(余弦曲线Cosine)##########
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)

##########控制时间，每秒控制15次##########
ctl_dt = 1 / 15

##########数据收集##########
scaler_q = defaultdict(list)
def smooth_dict(ori_dict):
    for k, v in ori_dict.items():
        scaler_q[k].append(float(v))

##########碰撞损失##########
def barrier(x: torch.Tensor, v_to_pt):
    return (v_to_pt * (1 - x).relu().pow(2)).mean()

##########动态保存策略##########
def is_save_iter(i):
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0


def is_save_trajectory_iter(i):
    return i == 0 or (i + 1) % 250 == 0


def rotation_matrix_to_rpy_deg(R):
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]
    r10 = R[..., 1, 0]
    r00 = R[..., 0, 0]

    pitch = torch.asin(torch.clamp(-r20, -1.0, 1.0))
    roll = torch.atan2(r21, r22)
    yaw = torch.atan2(r10, r00)
    return torch.stack([roll, pitch, yaw], dim=-1) * (180.0 / math.pi)


def build_yaw_frame(R):
    fwd = R[:, :, 0]
    zeros = torch.zeros_like(fwd)
    up = zeros.clone()
    up[:, 2] = 1.0
    fwd_h_raw = torch.stack([fwd[:, 0], fwd[:, 1], torch.zeros_like(fwd[:, 2])], dim=-1)
    fwd_h_norm = torch.norm(fwd_h_raw, 2, -1, keepdim=True)
    fallback = zeros.clone()
    fallback[:, 0] = 1.0
    fwd_h = torch.where(fwd_h_norm > 1e-6, fwd_h_raw / fwd_h_norm.clamp_min(1e-6), fallback)
    left = F.normalize(torch.cross(up, fwd_h, dim=-1), 2, -1, eps=1e-6)
    return torch.stack([fwd_h, left, up], -1)


def compute_heading_reference(env, R_yaw):
    target_vec = env.p_target - env.p.detach()
    zeros = torch.zeros_like(target_vec[:, 2])
    heading_ref_world = torch.stack([target_vec[:, 0], target_vec[:, 1], zeros], dim=-1)
    heading_norm = torch.norm(heading_ref_world, 2, -1, keepdim=True)
    fallback = R_yaw[:, :, 0]
    heading_ref_world = torch.where(
        heading_norm > 1e-6,
        heading_ref_world / heading_norm.clamp_min(1e-6),
        fallback,
    )
    heading_ref_local = torch.squeeze(heading_ref_world[:, None] @ R_yaw, 1)
    yaw_error = torch.atan2(heading_ref_local[:, 1], heading_ref_local[:, 0]).unsqueeze(-1)
    return heading_ref_world, heading_ref_local[:, :2], yaw_error


def decode_worker_action(act, R_yaw, yaw_rate_max_value):
    B_local = act.shape[0]
    act6 = act[:, :6]
    a_pred, v_pred = (R_yaw @ act6.reshape(B_local, 3, 2)).unbind(-1)
    yaw_rate_cmd = None
    if act.shape[-1] > 6:
        yaw_rate_cmd = torch.tanh(act[:, 6:7]) * float(yaw_rate_max_value)
    return a_pred, v_pred, yaw_rate_cmd


def _plotly_add_cuboid(fig, cx, cy, cz, hx, hy, hz, color='lightgray', opacity=0.68):
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    z0, z1 = cz - hz, cz + hz
    verts = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)
    tri = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=np.int32)
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
        color=color, opacity=opacity, flatshading=True, hoverinfo='skip', showlegend=False
    ))


def _plotly_add_sphere(fig, cx, cy, cz, r, color='royalblue', opacity=0.58, res_u=18, res_v=14):
    u = np.linspace(0.0, 2.0 * np.pi, res_u)
    v = np.linspace(0.0, np.pi, res_v)
    uu, vv = np.meshgrid(u, v)
    x = cx + r * np.cos(uu) * np.sin(vv)
    y = cy + r * np.sin(uu) * np.sin(vv)
    z = cz + r * np.cos(vv)
    fig.add_trace(go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=np.zeros_like(z),
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False,
        opacity=opacity,
        hoverinfo='skip',
        showlegend=False,
    ))


def _plotly_add_vertical_cylinder(fig, cx, cy, r, z0, z1, color='orange', opacity=0.52, res_theta=28, res_z=8):
    theta = np.linspace(0.0, 2.0 * np.pi, res_theta)
    z = np.linspace(z0, z1, res_z)
    tt, zz = np.meshgrid(theta, z)
    x = cx + r * np.cos(tt)
    y = cy + r * np.sin(tt)
    fig.add_trace(go.Surface(
        x=x,
        y=y,
        z=zz,
        surfacecolor=np.zeros_like(zz),
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False,
        opacity=opacity,
        hoverinfo='skip',
        showlegend=False,
    ))


def _is_outer_shell_voxel(env, cx, cy, cz, hx, hy, hz, eps=1e-2):
    x_max = float(getattr(env, 'map_x_max', 10.0))
    y_half = float(getattr(env, 'map_y_half', 12.0))
    y_min = float(getattr(env, 'map_y_min', -y_half))
    y_max = float(getattr(env, 'map_y_max', y_half))
    z_max = float(getattr(env, 'map_z_max', 5.0))
    boundary_half = float(getattr(env, 'boundary_half', 0.05))
    spawn_z_center = float(getattr(env, 'spawn_z_center', 2.5))
    spawn_x_center = float(getattr(env, 'spawn_x_center', x_max * 0.5))

    is_floor = abs(cz - 0.0) <= eps and abs(hz - boundary_half) <= eps
    is_ceiling = abs(cz - z_max) <= eps and abs(hz - boundary_half) <= eps

    is_x_side = (
        abs(hx - boundary_half) <= eps
        and abs(hy - y_half) <= eps
        and abs(cz - spawn_z_center) <= eps
        and (abs(cx - 0.0) <= eps or abs(cx - x_max) <= eps)
    )
    is_y_side = (
        abs(hy - boundary_half) <= eps
        and abs(hx - spawn_x_center) <= eps
        and abs(cz - spawn_z_center) <= eps
        and (abs(cy - y_min) <= eps or abs(cy - y_max) <= eps)
    )

    return is_floor, (is_ceiling or is_x_side or is_y_side)


def save_interactive_3d_html(html_path, env, p_cpu, v_cpu, R_cpu=None, idx=0, axis_len=0.25, axis_step=5):
    if go is None:
        return False

    traj_xyz = p_cpu.numpy()
    speed_cpu = v_cpu.norm(dim=-1).numpy()
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=traj_xyz[:, 0], y=traj_xyz[:, 1], z=traj_xyz[:, 2],
        mode='lines+markers',
        marker=dict(size=3, color=speed_cpu, colorscale='Turbo', colorbar=dict(title='Speed (m/s)')),
        line=dict(color='limegreen', width=5),
        name='Trajectory'
    ))

    if hasattr(env, 'voxels') and env.voxels.numel() > 0:
        vox = env.voxels[idx].detach().cpu().numpy()
        for box in vox[:180]:
            cx, cy, cz, hx, hy, hz = box.tolist()
            if hx > 20 or hy > 20 or hz > 20:
                continue
            is_floor, is_shell_not_floor = _is_outer_shell_voxel(env, cx, cy, cz, hx, hy, hz)
            if is_shell_not_floor:
                continue
            _plotly_add_cuboid(fig, cx, cy, cz, hx, hy, hz)

    if hasattr(env, 'balls') and env.balls.numel() > 0:
        balls = env.balls[idx].detach().cpu().numpy()
        for ball in balls:
            cx, cy, cz, r = ball.tolist()
            if r <= 1e-6:
                continue
            _plotly_add_sphere(fig, cx, cy, cz, r)

    if hasattr(env, 'cyl') and env.cyl.numel() > 0:
        cyls = env.cyl[idx].detach().cpu().numpy()
        z0 = 0.0
        z1 = float(getattr(env, 'map_z_max', 5.0))
        for cyl in cyls:
            cx, cy, r = cyl.tolist()
            if r <= 1e-6:
                continue
            _plotly_add_vertical_cylinder(fig, cx, cy, r, z0=z0, z1=z1)

    if R_cpu is not None:
        R_np = R_cpu.numpy()
        for t in range(0, len(traj_xyz), axis_step):
            pos = traj_xyz[t]
            Rm = R_np[t]
            for c, label, axis_i in [('red', 'X-axis', 0), ('green', 'Y-axis', 1), ('blue', 'Z-axis', 2)]:
                axis = Rm[:, axis_i] * axis_len
                fig.add_trace(go.Scatter3d(
                    x=[pos[0], pos[0] + axis[0]],
                    y=[pos[1], pos[1] + axis[1]],
                    z=[pos[2], pos[2] + axis[2]],
                    mode='lines',
                    line=dict(color=c, width=4),
                    showlegend=(t == 0),
                    name=label if t == 0 else None,
                ))

    fig.add_trace(go.Scatter3d(x=[traj_xyz[0, 0]], y=[traj_xyz[0, 1]], z=[traj_xyz[0, 2]], mode='markers',
                               marker=dict(size=6, color='green', symbol='circle'), name='Start'))
    fig.add_trace(go.Scatter3d(x=[traj_xyz[-1, 0]], y=[traj_xyz[-1, 1]], z=[traj_xyz[-1, 2]], mode='markers',
                               marker=dict(size=6, color='black', symbol='x'), name='End'))
    tgt = env.p_target[idx].detach().cpu().numpy()
    fig.add_trace(go.Scatter3d(x=[tgt[0]], y=[tgt[1]], z=[tgt[2]], mode='markers',
                               marker=dict(size=8, color='red', symbol='diamond'), name='Goal'))

    fig.update_layout(
        title='Interactive 3D Trajectory & Obstacles',
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        template='plotly_white',
        showlegend=True,
        margin=dict(l=5, r=5, t=40, b=5)
    )
    fig.write_html(html_path, include_plotlyjs='cdn')
    return True

##########pbar创建进度条，同时作为迭代器##########
pbar = tqdm(range(args.num_iters), ncols=120, miniters=max(1, int(args.terminal_log_interval)))
# depths = []
# states = []
B = args.batch_size
maze_update_counter = 0
for i in pbar:
    ######重置环境和模型######
    update_map = i == 0 or (
        maze_update_counter % max(1, int(args.maze_update_interval)) == 0
    )
    if args.use_precomputed_geometry_maps and update_map:
        current_precomputed_map_idx = precomputed_easy_indices[
            precomputed_map_update_count % len(precomputed_easy_indices)
        ]
        precomputed_map_update_count += 1
        current_precomputed_map_file = os.path.basename(
            precomputed_map_cache.map_files[current_precomputed_map_idx]
        )
        map_data = precomputed_map_cache.get_map(current_precomputed_map_idx)
        _align_env_goal_planes_to_precomputed_map(
            map_data,
            env,
            map_idx_hint=current_precomputed_map_idx,
        )
        env.current_map_idx = current_precomputed_map_idx
        env.reset_from_precomputed_map(map_data)
        map_msg = (
            f"iter={i + 1}, idx={current_precomputed_map_idx}, "
            f"type=easy, file={current_precomputed_map_file}"
        )
        print(f"[PrecomputedMap] {map_msg}")
        writer.add_text('Map/Current_Precomputed_File', map_msg, i + 1)
    elif args.use_precomputed_geometry_maps:
        env.reset_drone_only()
    elif update_map:
        env.reset()
    else:
        env.reset_drone_only()
    if not args.use_ball_obstacles:
        # Keep the source .pt cache immutable and filter only this training
        # environment. Empty [B, 0, 4] is supported by render/collision code.
        env.balls = env.balls[:, :0]
    maze_update_counter += 1
    writer.add_scalar(
        'Map/Precomputed_Enabled',
        1.0 if args.use_precomputed_geometry_maps else 0.0,
        i + 1,
    )
    if args.use_precomputed_geometry_maps:
        writer.add_scalar('Map/Current_Index', current_precomputed_map_idx, i + 1)
        writer.add_scalar('Map/Is_Easy', 1.0, i + 1)
    writer.add_scalar(
        'Map/Ball_Obstacles_Enabled',
        1.0 if args.use_ball_obstacles else 0.0,
        i + 1,
    )
    model.reset()
    capture_viz_now = (
        int(args.artifact_save_interval) > 0
        and (i + 1) % int(args.artifact_save_interval) == 0
    )
    ######初始化数据记录容器######
    p_history = []
    v_history = []
    R_history = []
    rpy_history = []
    act_cmd_history = []
    target_v_history = []
    yaw_rate_cmd_history = []
    yaw_error_history = []
    vec_to_pt_history = []
    act_diff_history = []
    v_preds = []
    depth_history = []
    v_net_feats = []
    h = None

    ######模拟控制延迟######
    act_lag = 1
    act_buffer = [env.act] * (act_lag + 1)
    yaw_rate_buffer = [torch.zeros((B, 1), device=device)] * (act_lag + 1)
    ######计算初始目标向量######
    target_v_raw = env.p_target - env.p
    ######偏航角角速度偏移######
    if args.yaw_drift:
        drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
        zeros = torch.zeros_like(drift_av)
        ones = torch.ones_like(drift_av)
        R_drift = torch.stack([
            torch.cos(drift_av), -torch.sin(drift_av), zeros,
            torch.sin(drift_av), torch.cos(drift_av), zeros,
            zeros, zeros, ones,
        ], -1).reshape(B, 3, 3)

    ######开始飞行仿真循环######
    for t in range(args.timesteps):####飞行总步长####
        ####模拟真实硬件控制时间误差####
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)
        ####生成视觉感知####
        depth, flow = env.render(ctl_dt)
        ####记录位置和最近障碍物向量####
        p_history.append(env.p)
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())

        ####按 mmgj 的统一保存周期采集第 0 架无人机深度帧####
        if capture_viz_now:
            depth_history.append(depth[0].detach())

        if args.yaw_drift:
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
        else:
            target_v_raw = env.p_target - env.p.detach()

        ####仿真器执行一个时间步####
        if use_attitude_v2:
            R_yaw_pre = build_yaw_frame(env.R)
            heading_ref_world_pre, _, _ = compute_heading_reference(env, R_yaw_pre)
            yaw_rate_step = yaw_rate_buffer[t]
            env.run(
                act_buffer[t], ctl_dt,
                heading_ref=heading_ref_world_pre,
                yaw_rate_cmd=yaw_rate_step,
                yaw_rate_max=yaw_rate_max,
            )
        else:
            env.run(act_buffer[t], ctl_dt, target_v_raw)

        ####构建航向旋转矩阵####
        ##去除了滚转（Roll）和俯仰（Pitch）”的纯偏航（Yaw）旋转矩阵##
        R = build_yaw_frame(env.R)

        ####计算理想参考速度####
        ##计算到目标的距离 target_v_raw 的模长##
        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        ##计算方向单位向量 (归一化)##
        target_v_unit = target_v_raw / target_v_norm
        # 速度限幅
        # 如果距离很远 (100m)，不要试图以 100m/s 飞过去，而是限制在 max_speed (比如 10m/s)。
        # 如果距离很近 (0.5m)，则速度就设为 0.5m/s (慢慢靠近)。
        target_v = target_v_unit * torch.clamp(target_v_norm, max=float(env.max_speed))
        ####组装喂给神经网络的“状态包”####
        state = [
            ##第一行相对目标向量##
            torch.squeeze(target_v[:, None] @ R, 1),
            ##机体 Z 轴在世界坐标系下的方向，代表重力向量 / 姿态感##
            env.R[:, 2],
            ##安全半径##
            env.margin[:, None]]
        if use_attitude_v2:
            heading_ref_world, heading_ref_local_xy, yaw_error = compute_heading_reference(env, R)
            yaw_rate_norm = getattr(env, "yaw_rate", torch.zeros((B, 1), device=device)) / float(yaw_rate_max)
            state.extend([heading_ref_local_xy, yaw_rate_norm])
        ####计算 无人机相对于自身机头方向的飞行速度####
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom:
            state.insert(0, local_v)
        ####把列表里的所有张量在最后一个维度拼接起来####
        state = torch.cat(state, -1)

        # normalize
        ####视觉预处理####
        x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
        ##最大值池化,长和宽都缩小 4 倍##
        x = F.max_pool2d(x[:, None], 4, 4)
        act, values, h = model(x, state, h)

        #神经网络预测加速度，预测速度；v2 额外预测 yaw_rate_cmd
        if use_attitude_v2:
            a_pred, v_pred, yaw_rate_cmd = decode_worker_action(act, R, yaw_rate_max)
        else:
            a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
            yaw_rate_cmd = None
        v_preds.append(v_pred)
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act)
        if use_attitude_v2:
            yaw_rate_used = yaw_rate_cmd
            yaw_rate_cmd_history.append(yaw_rate_cmd)
            yaw_rate_buffer.append(yaw_rate_used)
            yaw_error_history.append(yaw_error.detach())
        act_cmd_history.append(act_buffer[t])
        v_net_feats.append(torch.cat([act, local_v, h], -1))
        v_history.append(env.v)
        R_history.append(env.R)
        rpy_history.append(rotation_matrix_to_rpy_deg(env.R))
        target_v_history.append(target_v)

    ####轨迹位置历史####
    p_history = torch.stack(p_history)
    R_history = torch.stack(R_history)
    rpy_history = torch.stack(rpy_history)
    act_cmd_history = torch.stack(act_cmd_history)
    act_buffer = torch.stack(act_buffer)

    ####高度稳定损失：平滑跟踪起点高度到终点高度，避免中途掉高或上冲####
    altitude_progress = torch.linspace(
        0.0, 1.0, p_history.shape[0], device=device, dtype=p_history.dtype
    ).view(-1, 1)
    altitude_start = p_history[0, :, 2].detach()
    altitude_goal = env.p_target[:, 2].detach()
    altitude_reference = altitude_start.unsqueeze(0) + altitude_progress * (
        altitude_goal - altitude_start
    ).unsqueeze(0)
    loss_height_track = F.smooth_l1_loss(
        p_history[..., 2], altitude_reference
    )

    ####速度大小损失####
    v_history = torch.stack(v_history)
    v_history_cum = v_history.cumsum(0)#对速度累积求和
    v_history_avg = (v_history_cum[30:] - v_history_cum[:-30]) / 30 #最近30帧的平均速度
    target_v_history = torch.stack(target_v_history)
    T, B, _ = v_history.shape
    delta_v = torch.norm(v_history_avg - target_v_history[1:1-30], 2, -1)
    loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))

    ####速度预测误差####
    v_preds = torch.stack(v_preds)
    loss_v_pred = F.mse_loss(v_preds, v_history.detach())

    ####飞行偏离损失####
    ##强迫无人机把速度用在指向目标的方向，不要产生无用的横向漂移##
    target_v_history_norm = torch.norm(target_v_history, 2, -1)
    target_v_history_normalized = target_v_history / target_v_history_norm[..., None]
    fwd_v = torch.sum(v_history * target_v_history_normalized, -1)
    loss_bias = F.mse_loss(v_history, fwd_v[..., None] * target_v_history_normalized) * 3

    ####飞行平滑损失####
    jerk_history = act_buffer.diff(1, 0).mul(15)
    snap_history = F.normalize(act_buffer - env.g_std).diff(1, 0).diff(1, 0).mul(15**2)
    loss_d_acc = act_buffer.pow(2).sum(-1).mean()
    loss_d_jerk = jerk_history.pow(2).sum(-1).mean()
    loss_d_snap = snap_history.pow(2).sum(-1).mean()
    zero_loss = loss_d_acc.new_tensor(0.0)
    if use_attitude_v2 and len(yaw_rate_cmd_history) > 0:
        yaw_rate_cmd_tensor = torch.stack(yaw_rate_cmd_history)
        loss_yaw_cmd = zero_loss
        if yaw_rate_cmd_tensor.shape[0] > 1:
            loss_yaw_smooth = yaw_rate_cmd_tensor.diff(1, 0).pow(2).mean()
        else:
            loss_yaw_smooth = zero_loss
    else:
        loss_yaw_cmd = zero_loss
        loss_yaw_smooth = zero_loss

    ####避障损失和碰撞损失####
    vec_to_pt_history = torch.stack(vec_to_pt_history)
    distance = torch.norm(vec_to_pt_history, 2, -1)
    distance = distance - env.margin
    with torch.no_grad():
        v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
    loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)
    loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()

    ####纵向速度误差####
    ##朝着目标方向的有效速度达到一定值##
    speed_history = v_history.norm(2, -1)
    loss_speed = F.smooth_l1_loss(fwd_v, target_v_history_norm)

    loss = args.coef_v * loss_v + \
        args.coef_obj_avoidance * loss_obj_avoidance + \
        args.coef_bias * loss_bias + \
        args.coef_d_acc * loss_d_acc + \
        args.coef_d_jerk * loss_d_jerk + \
        args.coef_d_snap * loss_d_snap + \
        args.coef_speed * loss_speed + \
        args.coef_v_pred * loss_v_pred + \
        args.coef_collide * loss_collide + \
        args.coef_yaw_cmd * loss_yaw_cmd + \
        args.coef_yaw_smooth * loss_yaw_smooth + \
        args.coef_height_track * loss_height_track

    if torch.isnan(loss):
        print("loss is nan, exiting...")
        exit(1)

    pbar.set_description_str(f'loss: {loss:.3f}')
    optim.zero_grad()
    loss.backward()
    optim.step()
    sched.step()

    ######接下来运行的代码，不需要计算梯度######
    with torch.no_grad():
        avg_speed = speed_history.mean(0)
        if distance.dim() == 2:
            collision_free = torch.all(distance > 0, dim=0)
        else:
            collision_free = torch.all(distance.flatten(0, 1) > 0, dim=0)
        dist_to_goal = torch.norm(p_history - env.p_target.unsqueeze(0), 2, -1)
        final_dist_to_goal = dist_to_goal[-1]
        last_step_dist_to_goal = final_dist_to_goal.mean()
        if len(yaw_error_history) > 0:
            yaw_error_abs_deg = torch.stack(yaw_error_history).abs().mean() * (180.0 / math.pi)
            yaw_rate_env_abs_deg = getattr(env, "yaw_rate", torch.zeros((B, 1), device=device)).abs().mean() * (180.0 / math.pi)
        else:
            yaw_error_abs_deg = torch.tensor(0.0, device=device)
            yaw_rate_env_abs_deg = torch.tensor(0.0, device=device)
        reached_goal = torch.any(dist_to_goal < args.goal_radius, dim=0)
        success = collision_free & reached_goal
        _success = success.float().mean()
        _no_collision_rate = collision_free.float().mean()
        smooth_dict({
            'loss': loss,
            'loss_v': loss_v,
            'loss_v_pred': loss_v_pred,
            'loss_obj_avoidance': loss_obj_avoidance,
            'loss_d_acc': loss_d_acc,
            'loss_d_jerk': loss_d_jerk,
            'loss_d_snap': loss_d_snap,
            'loss_bias': loss_bias,
            'loss_speed': loss_speed,
            'loss_collide': loss_collide,
            'loss_height_track': loss_height_track,
            'loss_yaw_cmd': loss_yaw_cmd,
            'loss_yaw_smooth': loss_yaw_smooth,
            'success': _success,
            'no_collision_rate': _no_collision_rate,
            'reach_goal_rate': reached_goal.float().mean(),
            'final_dist_to_goal': final_dist_to_goal.mean(),
            'final_dist_to_goal_min': final_dist_to_goal.min(),
            'final_dist_to_goal_max': final_dist_to_goal.max(),
            'max_speed': speed_history.max(0).values.mean(),
            'avg_speed': avg_speed.mean(),
            'Heading/Yaw_Error_Abs_Deg': yaw_error_abs_deg,
            'Heading/Yaw_Rate_Abs_Deg': yaw_rate_env_abs_deg,
            'ar': (success.float() * avg_speed).mean()})
        # 逐轮记录“最后一个时间步到终点距离”，确保 TensorBoard 横轴为训练轮次（i+1）。
        writer.add_scalar('Metrics/LastStep_Dist_To_Goal', float(last_step_dist_to_goal.item()), i + 1)
        log_dict = {}
        if capture_viz_now:
            idx = 0
            depth_stack = (
                torch.stack(depth_history).detach().to(device='cpu', dtype=torch.float32)
                if depth_history else None
            )
            map_type = 'easy' if args.use_precomputed_geometry_maps else str(
                getattr(env, 'current_map_type', 'online') or 'online'
            )
            viz_record = {
                'iter': i + 1,
                'map_type': map_type,
                'map_idx': int(current_precomputed_map_idx),
                'map_file': str(current_precomputed_map_file),
                'stage': 'easy_only' if args.use_precomputed_geometry_maps else 'online',
                'env_snapshot': snapshot_env_for_viz(env, idx=idx),
                'p_cpu': p_history[:, idx].detach().cpu().clone(),
                'v_cpu': v_history[:, idx].detach().cpu().clone(),
                'rpy_cpu': rpy_history[:, idx].detach().cpu().clone(),
                'R_cpu': R_history[:, idx].detach().cpu().clone(),
                'act_cpu': act_cmd_history[:, idx].detach().cpu().clone(),
                'weights_cpu': None,
                'depth_stack': depth_stack,
                'astar_paths_sampled': [],
            }
            save_cached_viz_record(
                viz_record,
                i + 1,
                args=args,
                potential_map_cache=None,
                video_dir=video_dir,
                writer=writer,
            )
        if (i + 1) % 10000 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'checkpoint{i//10000:04d}.pth'))
        if (i + 1) % 25 == 0:
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)
            scaler_q.clear()
