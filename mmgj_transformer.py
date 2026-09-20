#9.2.7
# 07代码，增加proxy维度

import argparse
import math
from collections import defaultdict
import os
import datetime
import json
import sys
import matplotlib

matplotlib.use('Agg', force=True)

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print("[SDP] flash=False, mem_efficient=False, math=True (for higher-order gradients)")

try:
    from env_multi import Env
except ModuleNotFoundError:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from env_multi import Env
try:
    from potential_map_utils import PotentialMapCache, query_potential_guidance
except ModuleNotFoundError:
    PotentialMapCache = None
    query_potential_guidance = None
from env import probe_update_state_vec_common_upstream
from WorkNet_sru import WorkNet as SRUWorkNet
from WorkNet_transformer import WorkNet as TransformerWorkNet
from LossGenNet_transformer import LossGenNet
from worker_context_features import (
    WORKER_CONTEXT_FEATURE_DIM,
    extract_worker_context_features,
)
from utils.io_utils import (
    create_unique_experiment_dir,
    load_compatible_checkpoint,
    sync_multi_pub_to_checkpoint_dir,
)
from utils.logging_utils import (
    _diag_grad_meta,
    _diag_grad_tuple_to_params,
    _diag_output_to_params,
    _diag_output_to_params_count,
    _diag_should_log,
    _diag_tensor_finite,
    _grad_or_none_tuple,
    _resolve_map_log_key,
    _resolve_tb_writer,
    compute_meta_hypergrads_from_fast_grads,
    compute_gradient_alignment_loss,
    get_gradient_alignment_stats,
    get_grad_norm_from_grads,
    get_grad_stats,
    is_artifact_save_iter,
    merge_task_priority_gradients,
    release_autograd_graph_references,
    smooth_dict,
    summarize_temporal_weight_sensitivity,
)
from utils.map_utils import (
    _align_env_goal_planes_to_precomputed_map,
    _build_precomputed_map_type_indices,
    _precomputed_curriculum_stage,
    _select_precomputed_curriculum_map,
)
from utils.planner_utils import (
    GlobalPlanner,
    configure_planner_pool,
    compute_global_guidance_meta_loss,
)
from utils.rollout_utils import unrolled_meta_rollout
from utils.tensor_utils import (
    build_command_velocity,
    build_yaw_frame,
    compute_arrival_reward,
    compute_action_energy_loss_per_step,
    compute_action_smoothness_losses_per_step,
    compute_goal_progress_preference_loss,
    compute_heading_reference,
    compute_overlap_loss_per_step,
    compute_stuck_loss,
    compute_turn_preference_loss,
    compute_velocity_tracking_loss,
    compute_velocity_heading_command,
    decode_worker_action,
    extract_depth_geometry_features,
    extract_progress_features,
    rotation_matrix_to_rpy_deg,
    safe_l2_norm,
    sample_command_speed,
    select_min_clearance_obstacle_sample,
    sanitize_module_,
    sanitize_tensor,
)
from utils.visualization_utils import save_cached_viz_record, snapshot_env_for_viz


########### 1. 参数配置 ##########
parser = argparse.ArgumentParser()
parser.add_argument('--resume_worker', default="", help='Path to pretrained worker model')
parser.add_argument('--resume_lgn', default="", help='Path to pretrained lgn model')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_iters', type=int, default=20000)

# [优化策略参数]
parser.add_argument('--lgn_steps', type=int, default=1)
parser.add_argument('--worker_steps', type=int, default=1)

# 基础物理参数
parser.add_argument('--grad_decay', type=float, default=1.0,
                    help='Backward state-gradient retention per second; 1.0 disables artificial temporal decay')
parser.add_argument('--speed_mtp', type=float, default=1.0)
parser.add_argument('--command_speed_min', type=float, default=0.75,
                    help='Minimum random command speed before speed_mtp scaling')
parser.add_argument('--command_speed_max', type=float, default=3.25,
                    help='Maximum random command speed before speed_mtp scaling')
parser.add_argument('--command_speed', type=float, default=2.0,
                    help='Fixed command used by --no_random_command_speed, before speed_mtp scaling')
parser.add_argument('--random_command_speed', dest='random_command_speed', action='store_true',
                    help='Sample one official-style command speed per rollout/group (default)')
parser.add_argument('--no_random_command_speed', dest='random_command_speed', action='store_false')
parser.set_defaults(random_command_speed=True)
parser.add_argument('--goal_slowdown_time', type=float, default=1.0,
                    help='Seconds used for linear command-speed reduction near the goal')
parser.add_argument('--velocity_track_window', type=int, default=30,
                    help='Trailing actual-velocity averaging window')
parser.add_argument('--scene_scale', type=float, default=0.5,#調節環境大小
                    help='Global scene size scale for obstacle field extent and spawn area')
parser.add_argument('--obstacle_count_scale', type=float, default=0.5,#調節障礙物數量
                    help='Global obstacle-count scale; 0.5 matches the old precomputed easy-map spatial density')
parser.add_argument('--easy_density_scale', type=float, default=1.0,
                    help='Multiplier relative to the old precomputed easy-map spatial density')
parser.add_argument('--hard_density_scale', type=float, default=1.0,
                    help='Multiplier applied on top of the sampled hard density')
parser.add_argument('--easy_density_multiplier_min', type=float, default=1.0,
                    help='Minimum sampled easy density relative to the old easy-map density')
parser.add_argument('--easy_density_multiplier_max', type=float, default=1.0,
                    help='Maximum sampled easy density relative to the old easy-map density')
parser.add_argument('--hard_density_multiplier_min', type=float, default=1.0,
                    help='Minimum sampled hard density relative to the old easy-map density')
parser.add_argument('--hard_density_multiplier_max', type=float, default=2.0,
                    help='Maximum sampled hard density relative to the old easy-map density')
parser.add_argument('--start_goal_plane_y_abs', type=float, default=25,#調節起點和終點的位置
                    help='Start/goal planes are set to +Y and -Y using this absolute value')
parser.add_argument('--attitude_model', type=str, default='v2', choices=['legacy', 'v2'],
                    help='legacy uses goal-projected heading; v2 uses explicit yaw-rate dynamics')
parser.add_argument('--yaw_rate_max_deg', type=float, default=150.0)
parser.add_argument('--coef_yaw_cmd', type=float, default=0.2)
parser.add_argument('--coef_yaw_smooth', type=float, default=0.01)

# ===================== v2 机头跟踪参数（中文详解） =====================
# 这些参数只在 attitude_model='v2' 的显式 yaw 动力学下生效，用于控制：
# 1) 机头追踪真实速度方向的“转头力度”；
# 2) 是否允许网络输出的 yaw_rate 作为小幅残差；
# 3) 网络残差偏航与规则跟踪的组合方式。
#
# 调参建议（经验）：
# - 机头转得慢：增大 heading_yaw_kp 或 yaw_rate_max_deg。
# - 低速抖头：增大 heading_min_speed，或降低 heading_yaw_kp。
# - Worker 的 yaw-rate 残差会直接叠加到规则偏航率，再统一限幅。
parser.add_argument('--heading_track_mode', type=str, default='actual_v',
                    choices=['actual_v'],
                    help='Compatibility option; heading control tracks measured velocity (not the auxiliary prediction)')
parser.add_argument('--heading_min_speed', type=float, default=0.25,
                    help='Minimum horizontal speed reserved for velocity-heading control compatibility')
parser.add_argument('--heading_yaw_kp', type=float, default=4.0,
                    help='机头跟踪比例增益：yaw_rate_rule = heading_yaw_kp * yaw_error。越大转头越积极，但过大可能引起振荡')
parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
parser.add_argument('--timesteps', type=int, default=90)
parser.add_argument('--lgn_timesteps', type=int, default=90,
                    help='Rollout steps used in LGN phase; smaller value reduces 2nd-order gradient memory')
parser.add_argument('--exploration_time_window', type=int, default=1,
                    help='Look-back gap for exploration overlap loss; effective window is auto-clipped to keep valid long-range pairs')
parser.add_argument('--turn_speed_threshold', type=float, default=0.2,
                    help='Center speed (m/s) of the differentiable low-speed gate for turn loss')
parser.add_argument('--turn_speed_softness', type=float, default=0.01,
                    help='Sigmoid transition width (m/s) of the low-speed gate for turn loss')
parser.add_argument('--turn_soft_angle_deg', type=float, default=10.0,
                    help='Quadratic-to-linear angle boundary (degrees) for 3D turn loss')
parser.add_argument('--speed_change_soft_delta', type=float, default=0.05,
                    help='Deprecated compatibility option for the LGN turn preference')
parser.add_argument('--detach_interval', type=int, default=12,
                    help='Detach temporal memory every N steps to limit graph depth (<=0 disables)')
parser.add_argument('--cam_angle', type=int, default=10)
parser.add_argument('--goal_radius', type=float, default=1.0,
                    help='Episode terminates when all drones are within this radius of their goal')
parser.add_argument('--meta_arrival_reward_radius', type=float, default=0.5,
                    help='Goal-ball radius where arrival reward starts reducing meta loss')
parser.add_argument('--arrival_soft_temperature', type=float, default=0.05,
                    help='Sigmoid temperature (m) for differentiable first-arrival detection')
parser.add_argument('--meta_arrival_reward_weight', type=float, default=1.0,
                    help='Weight of the arrival reward subtracted from meta loss')
parser.add_argument('--maze_update_interval', type=int, default=10,
                    help='Generate a new random maze online every N iterations; drone-only reset in between')

# 时序记忆参数
parser.add_argument('--worker_backbone', type=str, default='sru',
                    choices=['sru', 'transformer'],
                    help='Worker temporal backbone; this SRU branch defaults to spatial recurrent memory')
parser.add_argument('--sru_hidden_channels', type=int, default=96,
                    help='Channel count of the SRU spatial memory')
parser.add_argument('--worker_max_seq_len', type=int, default=32,
                    help='Maximum token memory length when --worker_backbone=transformer')
parser.add_argument('--lgn_max_seq_len', type=int, default=32,
                    help='Maximum raw-token memory length for the causal Transformer LGN')
parser.add_argument('--activation_checkpoint', dest='activation_checkpoint', action='store_true',
                    help='Enable activation checkpointing for the Worker and Transformer LGN')
parser.add_argument('--no_activation_checkpoint', dest='activation_checkpoint', action='store_false',
                    help='Disable Transformer activation checkpointing for Worker and LGN (default)')
parser.set_defaults(activation_checkpoint=False)

# 环境Flag
parser.add_argument('--single', default=True, action='store_true')
parser.add_argument('--gate', default=False, action='store_true')
parser.add_argument('--ground_voxels', default=False, action='store_true')
parser.add_argument('--scaffold', default=False, action='store_true')
parser.add_argument('--random_rotation', default=False, action='store_true')
parser.add_argument('--no_odom', default=False, action='store_true')
# [开关1] U 型局部最优陷阱地图开关（默认: 关闭）
# - 默认行为：不传任何参数时 include_u_local_optimum=False。
# - 显式开启：--include_u_local_optimum
# - 关闭陷阱：--no_include_u_local_optimum（地图为 hard/easy/easy 三块随机重排，且不再包含 U 区）
# - 说明：这两个参数写在同一行时，以最后一个为准（argparse store_true/store_false 同目标变量）。
parser.add_argument('--include_u_local_optimum', dest='include_u_local_optimum', action='store_true',
                    help='Include U-shaped local-optimum trap region in three-zone map')
parser.add_argument('--no_include_u_local_optimum', dest='include_u_local_optimum', action='store_false',
                    help='Disable U-shaped trap region and use shuffled hard/easy/easy region order')
parser.set_defaults(include_u_local_optimum=False)
# [开关1.1] 两分区紧凑地图开关（默认: 关闭）
# - 默认行为：不传任何参数时 compact_two_zone_map=False，保持当前三分区地图逻辑不变。
# - 显式开启：--compact_two_zone_map（仅保留 easy/hard 两种地图，Y 向尺寸缩小，起终点平面随之调整）
# - 显式关闭：--no_compact_two_zone_map
# - 说明：与 include_u_local_optimum 共存时，开启紧凑两分区会优先使用两分区布局（不再包含 U 区）。
parser.add_argument('--compact_two_zone_map', dest='compact_two_zone_map', action='store_true',
                    help='Use compact two-zone map (easy+hard only), with smaller map and adjusted start/goal planes')
parser.add_argument('--no_compact_two_zone_map', dest='compact_two_zone_map', action='store_false',
                    help='Use default map layout (current behavior)')
parser.set_defaults(compact_two_zone_map=True)
# [开关2] 墙壁物理反馈开关（默认: 关闭）
# - 默认行为：不传任何参数时 wall_physical_feedback=False，采用自由运动结果（当前代码行为）。
# - 开启反馈：--wall_physical_feedback（启用软接触反馈，修正穿墙/贴墙时的位置与速度）
# - 显式关闭：--no_wall_physical_feedback
# - 典型组合：
#   1) 保持当前基线：不加这两个开关（或显式 --include_u_local_optimum --no_wall_physical_feedback）
#   2) 仅去掉 U 陷阱：--no_include_u_local_optimum
#   3) 仅加墙体反馈：--wall_physical_feedback
#   4) 同时去陷阱+加反馈：--no_include_u_local_optimum --wall_physical_feedback
parser.add_argument('--wall_physical_feedback', dest='wall_physical_feedback', action='store_true',
                    help='Enable wall-contact physical feedback correction in env dynamics')
parser.add_argument('--no_wall_physical_feedback', dest='wall_physical_feedback', action='store_false',
                    help='Disable wall-contact physical feedback and use free-motion result')
parser.set_defaults(wall_physical_feedback=False)

# 学习率
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--lgn_lr', type=float, default=2e-4)
parser.add_argument('--inner_lr', type=float, default=5e-4,
                    help='Inner loop LR for differentiable worker update in LGN phase')
parser.add_argument('--inner_steps', type=int, default=1,
                    help='Number of differentiable inner SGD steps (unrolled bilevel)')
parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                    help='Gradient-norm cap for virtual inner-loop Worker and LGN updates')
parser.add_argument('--worker_grad_clip_norm', type=float, default=5.0,
                    help='Final global gradient-norm cap for the persistent Worker after task-priority gradient merging')
parser.add_argument('--worker_proxy_weight', type=float, default=0.5,
                    help='Base scale of the LGN proxy objective in persistent and virtual Worker updates')
parser.add_argument('--worker_proxy_grad_ratio', type=float, default=0.5,
                    help='Maximum proxy-gradient norm relative to the direct arrival-task gradient in persistent Worker updates')
parser.add_argument('--worker_arrival_weight', type=float, default=1.0,
                    help='Weight of direct terminal-distance and arrival-reward loss in persistent and virtual Worker updates')
parser.add_argument('--worker_velocity_track_weight', type=float, default=1.0,
                    help='Fixed Worker weight for actual command-velocity tracking')
parser.add_argument('--worker_velocity_predict_weight', type=float, default=2.0,
                    help='Fixed Worker weight for velocity prediction')
parser.add_argument('--worker_collision_weight', type=float, default=2.0,
                    help='Fixed Worker collision-safety weight')
parser.add_argument('--meta_velocity_track_weight', type=float, default=1.0,
                    help='Outer-meta weight for command-velocity compliance')
parser.add_argument('--lgn_grad_alignment_weight', type=float, default=0.1,
                    help='Weight of LGN auxiliary loss aligning dynamic proxy and unrolled-meta Worker gradients')
parser.add_argument('--exp_name', type=str, default="default", help="Extra tag for experiment")
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                    help='Experiment output parent; relative paths are resolved inside this project copy')

# 避障/碰撞超参
parser.add_argument('--avoid_safe_margin', type=float, default=0.35,
                    help='Proxy avoidance rises smoothly inside this clearance to walls')
parser.add_argument('--lgn_output_temperature', type=float, default=1.0,
                    help='Temperature of the smoothly bounded softplus used for LGN loss weights')
parser.add_argument('--lgn_weight_floor', type=float, default=0.01,
                    help='Strict lower bound for Avoidance/Exploration/Turn LGN weights')
parser.add_argument('--lgn_preference_weight_floor', type=float, default=0.01,
                    help='Lower bound for Progress/Smoothness/Energy weights; defaults to the same 0.01 floor as safety weights')
parser.add_argument('--lgn_weight_ceiling', type=float, default=5.0,
                    help='Strict upper bound for all learned LGN loss weights')
parser.add_argument('--resume_lgn_legacy_six_output', default=False, action='store_true',
                    help='Interpret a six-row LGN checkpoint as legacy Speed/Direction/Avoid/Explore/Turn layout')
parser.add_argument('--speed_goal_slow_dist', type=float, default=2.5,
                    help='Deprecated compatibility option; LGN no longer outputs a target speed')
parser.add_argument('--meta_coll_soft_weight', type=float, default=5.0,
                    help='Soft collision term weight in meta loss')
parser.add_argument('--meta_coll_hard_weight', type=float, default=40.0,
                    help='Hard penetration-depth penalty weight in meta loss')
parser.add_argument('--meta_coll_event_weight', type=float, default=80.0,
                    help='Episode-level collision event penalty weight in meta loss')
parser.add_argument('--meta_coll_event_temp', type=float, default=80.0,
                    help='Sharpness for differentiable episode collision event penalty (sigmoid temperature)')
parser.add_argument('--meta_coll_event_threshold', type=float, default=0.01,
                    help='Penetration-depth threshold (m) where differentiable collision-event penalty turns on')
parser.add_argument('--speed_near_obs_floor', type=float, default=0.05,
                    help='Deprecated compatibility option; LGN no longer outputs a target speed')
parser.add_argument('--stuck_loss_weight', type=float, default=2.0,
                    help='Weight for local displacement-based stuck penalty')
parser.add_argument('--stuck_window', type=int, default=15,
                    help='Window size for stuck detection (steps)')
parser.add_argument('--stuck_displacement_threshold', type=float, default=0.3,
                    help='Minimum displacement in window before stuck penalty activates (m)')
parser.add_argument('--collision_duration_weight', type=float, default=10.0,
                    help='Weight for collision-duration diagnostic penalty')
parser.add_argument('--meta_smooth_jerk_weight', type=float, default=0.001,
                    help='Meta loss weight for first-order action difference (jerk) smoothing')
parser.add_argument('--meta_smooth_snap_weight', type=float, default=0.0002,
                    help='Meta loss weight for second-order normalized action difference (snap) smoothing')
parser.add_argument('--meta_progress_weight', type=float, default=0.5,
                    help='Unrolled/main meta weight for bounded per-step goal progress preference')
parser.add_argument('--meta_energy_weight', type=float, default=0.05,
                    help='Unrolled/main meta weight for normalized action-energy preference')
parser.add_argument('--energy_action_scale', type=float, default=10.0,
                    help='Acceleration-command scale used to normalize the energy preference loss')

# 全局规划引导元损失参数
parser.add_argument('--meta_guidance_weight', type=float, default=0.5,
                    help='Weight for global guidance meta loss (path-guiding dense supervision)')
parser.add_argument('--guide_sample_count', type=int, default=10,
                    help='Number of keypoints to sample for guidance loss computation')
parser.add_argument('--guide_sample_strategy', type=str, default='random',
                    choices=['random', 'uniform', 'adaptive', 'critical'],
                    help='Sampling strategy: random/uniform/adaptive(danger+curvature)/critical(start+end+danger)')
parser.add_argument('--guide_max_accel', type=float, default=5.0,
                    help='Max acceleration for trapezoidal velocity profile (m/s^2)')
parser.add_argument('--guide_max_decel', type=float, default=6.0,
                    help='Max deceleration for trapezoidal velocity profile (m/s^2)')
parser.add_argument('--guide_dir_weight', type=float, default=0.5,
                    help='Weight for direction alignment loss within guidance loss')
parser.add_argument('--guide_speed_weight', type=float, default=0.3,
                    help='Weight for overspeed penalty within guidance loss')
parser.add_argument('--guide_lateral_weight', type=float, default=0.3,
                    help='Weight for lateral error penalty (geometric distance to planned path)')
parser.add_argument('--guide_speed_diff_weight', type=float, default=0.2,
                    help='Weight for speed difference penalty (overspeed + underspeed)')
parser.add_argument('--guide_escape_weight', type=float, default=1.0,
                    help='Weight for escape penalty on collided points')
parser.add_argument('--guide_recovery_speed_weight', type=float, default=0.15,
                    help='Extra speed damping weight on planner-invalid sampled points')
parser.add_argument('--guide_collision_threshold', type=float, default=-0.05,
                    help='Penetration threshold below which point is considered collided')
parser.add_argument('--guide_accel_weight', type=float, default=0.1,
                    help='Weight for acceleration mismatch penalty (deceleration requirement)')
parser.add_argument('--planner_resolution', type=float, default=0.3,
                    help='Resolution of the occupancy grid for A* planning (meters)')
parser.add_argument('--planner_margin', type=float, default=0.15,
                    help='Safety margin for obstacle inflation in planner (meters)')
parser.add_argument('--planner_parallel', dest='planner_parallel', action='store_true',
                    help='Enable sample-level parallel global planning with multiprocessing pool')
parser.add_argument('--no_planner_parallel', dest='planner_parallel', action='store_false',
                    help='Disable sample-level parallel global planning')
parser.set_defaults(planner_parallel=True)
parser.add_argument('--planner_workers', type=int, default=0,
                    help='Number of planner worker processes (<=0 means auto)')
parser.add_argument('--planner_pool_maxtasks', type=int, default=256,
                    help='maxtasksperchild for planner process pool to avoid long-run memory growth')
# [引导后端切换开关]
# - none: 默认在线生成随机障碍物，不计算全局规划/势场引导
# - astar: 使用在线 A* 规划引导（原有方案）
# - dijkstra_potential: 使用离线缓存 Dijkstra 势场引导
# 使用方法：
# - 默认不写时为 none（在线随机地图）。
# - 势场模式默认读取双倍障碍物复杂度、3x3 平铺的 easy 地图缓存。
#   需要配合预计算地图目录，例如：
#   --precomputed_map_dir /home/robot/transformer/precomputed_maps_turn_encouragement_density2x_tiled3x3 --num_precomputed_maps 0
# - 切回 A* 模式：--guidance_backend astar
# 说明：这是统一开关，优先于旧的 use_precomputed_potential_maps/use_astar_guidance 组合语义。
parser.add_argument('--guidance_backend', type=str, default='none',
                    choices=['none', 'astar', 'dijkstra_potential'],
                    help='Global guidance backend; none uses online random maps by default')
parser.add_argument('--use_precomputed_potential_maps', default=False, action='store_true',
                    help='Use precomputed Dijkstra potential-map guidance instead of online A* planning')
parser.add_argument('--precomputed_map_dir', type=str, default='/home/robot/transformer/precomputed_maps_turn_encouragement_density2x_tiled3x3',
                    help='Directory containing full precomputed potential cache .pt files')
parser.add_argument('--precomputed_geometry_map_dir', type=str,
                    default='/home/robot/transformer/precomputed_geometry_density2x_tiled3x3',
                    help='Directory containing geometry-only map files used when potential guidance is disabled')
parser.add_argument('--use_precomputed_geometry_maps', dest='use_precomputed_geometry_maps', action='store_true',
                    help='Load pre-generated obstacle geometry independently of the guidance backend')
parser.add_argument('--no_precomputed_geometry_maps', dest='use_precomputed_geometry_maps', action='store_false',
                    help='Disable pre-generated geometry and return to online environment generation')
parser.set_defaults(use_precomputed_geometry_maps=False)
parser.add_argument('--num_precomputed_maps', type=int, default=0,
                    help='Max number of precomputed maps to load from precomputed_map_dir (<=0 means all)')
# [势场查询参数]
# - trilinear: 三线性插值，点落在栅格内部时按8个角点加权，训练更平滑（推荐）
# - nearest: 最近邻查询，调试方便但梯度更离散
parser.add_argument('--potential_interpolation', type=str, default='trilinear',
                    choices=['nearest', 'trilinear'],
                    help='Interpolation mode for querying potential/vector field at continuous positions')
# [势场下降约束参数]
# ReLU(phi[t+1]-phi[t]+delta) 中的 delta。
# 取负值(如 -0.01)表示“允许极小上升噪声，但整体应下降”。
parser.add_argument('--potential_delta_margin', type=float, default=-0.01,
                    help='Delta margin in potential decrease loss: ReLU(phi[t+1]-phi[t]+delta)')
parser.add_argument('--use_astar_guidance', default=False, action='store_true',
                    help='Force legacy online A* guidance even when precomputed maps are enabled')
parser.add_argument('--guidance_all_phases', dest='guidance_all_phases', action='store_true',
                    help='Compute guidance loss/metrics in worker phase too (A* backend may be slower)')
parser.add_argument('--no_guidance_all_phases', dest='guidance_all_phases', action='store_false',
                    help='Compute guidance only in LGN phase unless potential backend auto-enables worker phase')
parser.set_defaults(guidance_all_phases=False)
parser.add_argument('--diag_interval', type=int, default=100,
                    help='Print detailed DIAG logs every N iterations (<=0 disables)')
parser.add_argument('--diag_second_order', dest='diag_second_order', action='store_true',
                    help='Enable heavy second-order diagnostic probes (can be noisy and slow)')
parser.add_argument('--no_diag_second_order', dest='diag_second_order', action='store_false',
                    help='Disable heavy second-order diagnostic probes')
parser.set_defaults(diag_second_order=False)
parser.add_argument('--terminal_log_interval', type=int, default=500,
                    help='Update terminal progress/log text every N iterations')
parser.add_argument('--debug_scalar_interval', type=int, default=25,
                    help='Unified TensorBoard scalar logging interval in iterations (<=0 disables periodic scalar writes)')
parser.add_argument('--debug_tb_interval', type=int, default=500,
                    help='TensorBoard Debug/* logging interval in iterations (<=0 disables Debug tag writes)')
parser.add_argument('--sparse_temporal_monitor_interval', type=int, default=500,
                    help='Probe sparse-arrival sensitivity across early/middle/late LGN timesteps every N iterations (<=0 disables)')
parser.add_argument('--artifact_save_interval', type=int, default=1000,
                    help='Unified checkpoint, trajectory, and video save interval (<=0 disables periodic saves)')
parser.add_argument('--trajectory_save_interval', type=int, default=None,
                    help='Deprecated alias for --artifact_save_interval')

args = parser.parse_args()
if args.trajectory_save_interval is not None:
    args.artifact_save_interval = int(args.trajectory_save_interval)
if args.energy_action_scale <= 0.0:
    parser.error('--energy_action_scale must be positive')
if args.meta_progress_weight < 0.0 or args.meta_energy_weight < 0.0:
    parser.error('--meta_progress_weight and --meta_energy_weight must be non-negative')
if not math.isfinite(args.command_speed_min) or args.command_speed_min < 0.0:
    parser.error('--command_speed_min must be finite and non-negative')
if not math.isfinite(args.command_speed_max) or args.command_speed_max < args.command_speed_min:
    parser.error('--command_speed_max must be finite and >= --command_speed_min')
if not math.isfinite(args.command_speed) or args.command_speed < 0.0:
    parser.error('--command_speed must be finite and non-negative')
if args.goal_slowdown_time <= 0.0 or args.velocity_track_window <= 0:
    parser.error('--goal_slowdown_time and --velocity_track_window must be positive')
if args.sru_hidden_channels < 8:
    parser.error('--sru_hidden_channels must be at least 8')
for fixed_weight_name in (
    'worker_velocity_track_weight', 'worker_velocity_predict_weight',
    'worker_collision_weight', 'meta_velocity_track_weight',
):
    fixed_weight = getattr(args, fixed_weight_name)
    if not math.isfinite(fixed_weight) or fixed_weight < 0.0:
        parser.error(f'--{fixed_weight_name} must be finite and non-negative')
if args.maze_update_interval <= 0:
    parser.error('--maze_update_interval must be positive')
for difficulty in ('easy', 'hard'):
    density_min = getattr(args, f'{difficulty}_density_multiplier_min')
    density_max = getattr(args, f'{difficulty}_density_multiplier_max')
    if not math.isfinite(density_min) or density_min <= 0.0:
        parser.error(f'--{difficulty}_density_multiplier_min must be finite and positive')
    if not math.isfinite(density_max) or density_max < density_min:
        parser.error(f'--{difficulty}_density_multiplier_max must be finite and >= minimum')
yaw_rate_max = math.radians(float(args.yaw_rate_max_deg))
use_attitude_v2 = args.attitude_model == 'v2'

# 默认在线生成地图；可显式启用预生成几何或 Dijkstra 势场缓存。
args.guidance_enabled = args.guidance_backend != 'none'
args.use_precomputed_potential_maps = args.guidance_backend == 'dijkstra_potential'
args.use_astar_guidance = args.guidance_backend == 'astar'
args.use_precomputed_geometry_maps = bool(
    args.use_precomputed_geometry_maps or args.use_precomputed_potential_maps
)

# 势场后端查询代价较低，默认在 worker phase 也计算 guidance 指标/损失。
args.guidance_all_phases = bool(
    args.guidance_enabled
    and (args.guidance_all_phases or args.use_precomputed_potential_maps)
)

# Planner parallel runtime config (used by guidance reference computation)
configure_planner_pool(
    enabled=bool(args.guidance_enabled and args.planner_parallel),
    num_workers=args.planner_workers,
    maxtasks_per_child=args.planner_pool_maxtasks,
)
POTENTIAL_MAP_CACHE = None

########## 2. 目录与日志初始化 ##########
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
script_name = os.path.splitext(os.path.basename(__file__))[0]
save_dir_name = "07代码，增加proxy维度"
checkpoint_parent = args.checkpoint_dir
if not os.path.isabs(checkpoint_parent):
    checkpoint_parent = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        checkpoint_parent,
    )
save_dir = create_unique_experiment_dir(
    checkpoint_parent,
    save_dir_name,
)
video_dir = os.path.join(save_dir, 'videos')

os.makedirs(video_dir, exist_ok=True)
print(f"Training artifacts will be saved to: {save_dir}")

with open(os.path.join(save_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

# Keep one immutable source snapshot per experiment.  Re-syncing the whole
# workspace at every artifact interval is expensive and makes the snapshot
# depend on files edited while training is already running.
try:
    sync_stats = sync_multi_pub_to_checkpoint_dir(save_dir)
    print(
        f"[CodeSnapshot] src={sync_stats['src_root']} -> dst={sync_stats['dst_root']} "
        f"copied={sync_stats['files_copied']}"
    )
except Exception as e:
    print(f"[CodeSnapshot][WARN] initial source snapshot failed: {e}")

writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
map_writers = {}
print(f"[TensorBoard] Unified log directory: {os.path.join(save_dir, 'logs')}")


device = torch.device('cuda')

########## 3. 环境初始化 ##########
env = Env(args.batch_size, 64, 48, args.grad_decay, device,
          fov_x_half_tan=args.fov_x_half_tan, single=args.single,
          gate=args.gate, ground_voxels=args.ground_voxels,
          scaffold=args.scaffold, speed_mtp=args.speed_mtp,
          scene_scale=args.scene_scale,
          random_rotation=args.random_rotation, cam_angle=args.cam_angle,
          obstacle_count_scale=args.obstacle_count_scale,
          easy_density_scale=args.easy_density_scale,
          hard_density_scale=args.hard_density_scale,
          easy_density_multiplier_min=args.easy_density_multiplier_min,
          easy_density_multiplier_max=args.easy_density_multiplier_max,
          hard_density_multiplier_min=args.hard_density_multiplier_min,
          hard_density_multiplier_max=args.hard_density_multiplier_max,
          start_goal_plane_y_abs=args.start_goal_plane_y_abs,
          include_u_local_optimum=args.include_u_local_optimum,
          compact_two_zone_map=args.compact_two_zone_map,
          wall_physical_feedback=args.wall_physical_feedback)


TRAIN_MAP_TYPES = ("easy",)
PRECOMPUTED_CURRICULUM_REQUIRED_TYPES = TRAIN_MAP_TYPES
VIZ_MAP_TYPES = TRAIN_MAP_TYPES if args.use_precomputed_geometry_maps else ("random",)
PRECOMPUTED_MAP_TYPE_CODES = {
    "none": -1,
    "easy": 0,
    "hairpin": 1,
    "u_min": 2,
    "hard": 3,
    "legacy": 4,
    "random": 5,
}
PRECOMPUTED_CURRICULUM_STAGE_CODES = {
    "none": -1,
    "easy_only": 0,
    "easy_hairpin": 1,
    "easy_hairpin_u_min": 2,
    "online_random": 3,
}


PRECOMPUTED_MAP_TYPE_INDICES = {}
INITIAL_PRECOMPUTED_MAP_IDX = -1
INITIAL_PRECOMPUTED_MAP_TYPE = "none"
INITIAL_PRECOMPUTED_MAP_FILE = ""

if args.use_precomputed_geometry_maps:
    if PotentialMapCache is None:
        raise RuntimeError("potential_map_utils.py is required for precomputed map loading")
    if args.use_precomputed_potential_maps and query_potential_guidance is None:
        raise RuntimeError("query_potential_guidance is required for dijkstra_potential guidance")
    active_map_dir = (
        args.precomputed_map_dir
        if args.use_precomputed_potential_maps
        else args.precomputed_geometry_map_dir
    )
    POTENTIAL_MAP_CACHE = PotentialMapCache(
        map_dir=active_map_dir,
        num_maps=args.num_precomputed_maps,
    )
    if len(POTENTIAL_MAP_CACHE) <= 0:
        raise RuntimeError(
            f"No precomputed maps found in {active_map_dir}."
        )
    PRECOMPUTED_MAP_TYPE_INDICES = _build_precomputed_map_type_indices(POTENTIAL_MAP_CACHE)
    missing_types = [
        map_type for map_type in PRECOMPUTED_CURRICULUM_REQUIRED_TYPES
        if len(PRECOMPUTED_MAP_TYPE_INDICES.get(map_type, [])) == 0
    ]
    if missing_types:
        counts_msg = ", ".join(
            f"{k}={len(v)}" for k, v in sorted(PRECOMPUTED_MAP_TYPE_INDICES.items())
        )
        raise RuntimeError(
            "Easy-only training requires at least one easy precomputed map. "
            f"Missing: {','.join(missing_types)}. loaded_counts: {counts_msg}. "
            "If --num_precomputed_maps is set, increase it or use 0 to load all maps."
        )

    INITIAL_PRECOMPUTED_MAP_TYPE = "easy"
    INITIAL_PRECOMPUTED_MAP_IDX = PRECOMPUTED_MAP_TYPE_INDICES[INITIAL_PRECOMPUTED_MAP_TYPE][0]
    INITIAL_PRECOMPUTED_MAP_FILE = os.path.basename(POTENTIAL_MAP_CACHE.map_files[INITIAL_PRECOMPUTED_MAP_IDX])
    first_map = POTENTIAL_MAP_CACHE.get_map(INITIAL_PRECOMPUTED_MAP_IDX)
    _align_env_goal_planes_to_precomputed_map(first_map, env, map_idx_hint=INITIAL_PRECOMPUTED_MAP_IDX)
    env.reset_from_precomputed_map(first_map)
    env.current_map_idx = INITIAL_PRECOMPUTED_MAP_IDX
    counts_msg = ", ".join(
        f"{k}={len(v)}" for k, v in sorted(PRECOMPUTED_MAP_TYPE_INDICES.items())
    )
    print(
        f"[PrecomputedMap] loaded={len(POTENTIAL_MAP_CACHE)} "
        f"from {active_map_dir}, initial={INITIAL_PRECOMPUTED_MAP_FILE}, "
        f"counts=({counts_msg}), guidance_backend={args.guidance_backend}"
    )
else:
    print(
        f"[OnlineMap] Regenerate every {args.maze_update_interval} iterations; "
        f"easy density=[{args.easy_density_multiplier_min:.2f}, {args.easy_density_multiplier_max:.2f}]x, "
        f"hard density=[{args.hard_density_multiplier_min:.2f}, {args.hard_density_multiplier_max:.2f}]x; "
        f"obstacle bounds x=[{env.obstacle_x_min:.1f}, {env.obstacle_x_max:.1f}], "
        f"y=[{env.obstacle_y_min:.1f}, {env.obstacle_y_max:.1f}]"
    )

_upstream_probe = probe_update_state_vec_common_upstream(device)
env.update_state_vec_in_meta_path = bool(_upstream_probe["is_common_upstream"])
print(
    f"[Phase1 Probe] update_state_vec common-upstream="
    f"{env.update_state_vec_in_meta_path} (delta={_upstream_probe['delta']:.6g})"
)

base_state_dim = 7 if args.no_odom else 10
state_dim = base_state_dim + (3 if use_attitude_v2 else 0)
action_dim = 7 if use_attitude_v2 else 6
geom_dim = 19
progress_dim = 8
progress_dim += WORKER_CONTEXT_FEATURE_DIM
worker_state_dim = state_dim + geom_dim + progress_dim

if args.worker_backbone == 'sru':
    worknet = SRUWorkNet(
        worker_state_dim,
        action_dim,
        max_seq_len=args.worker_max_seq_len,
        activation_checkpoint=args.activation_checkpoint,
        hidden_channels=args.sru_hidden_channels,
    )
else:
    worknet = TransformerWorkNet(
        worker_state_dim,
        action_dim,
        max_seq_len=args.worker_max_seq_len,
        activation_checkpoint=args.activation_checkpoint,
    )
worknet = worknet.to(device)
print(
    f"[Worker] backbone={args.worker_backbone}, "
    f"parameters={sum(parameter.numel() for parameter in worknet.parameters()):,}"
)

lgn = LossGenNet(
    state_dim=state_dim,
    geom_dim=geom_dim,
    progress_dim=progress_dim,
    max_seq_len=args.lgn_max_seq_len,
    output_temperature=args.lgn_output_temperature,
    weight_floor=args.lgn_weight_floor,
    preference_weight_floor=args.lgn_preference_weight_floor,
    weight_ceiling=args.lgn_weight_ceiling,
    activation_checkpoint=args.activation_checkpoint,
).to(device)
########## 4. 加载预训练模型 ##########


load_compatible_checkpoint(
    worknet,
    args.resume_worker,
    "Worker",
    device,
    zero_expanded=True,
    output_row_mappings=(
        {4: [0, None, 1, None, 2, None, 3], 3: [0, None, 1, None, 2, None, None]}
        if use_attitude_v2
        else {3: [0, None, 1, None, 2, None]}
    ),
)
# Preserve only the learned Avoidance/Exploration/Turn rows when loading an
# older LGN whose leading rows represented speed or direction outputs.
load_compatible_checkpoint(
    lgn,
    args.resume_lgn,
    "LGN",
    device,
    zero_expanded=False,
    output_row_mappings={
        3: [0, 1, 2, None, None, None],
        5: [2, 3, 4, None, None, None],
        6: [3, 4, 5, None, None, None],
        9: [6, 7, 8, None, None, None],
    },
    force_output_row_mapping=bool(args.resume_lgn_legacy_six_output),
)

########## 5. 优化器配置 ##########
optim_worker = AdamW(worknet.parameters(), args.lr)
optim_lgn = AdamW(lgn.parameters(), args.lgn_lr)
sched = CosineAnnealingLR(optim_worker, args.num_iters, args.lr * 0.01)

########## 6. 日志状态 ##########
scaler_q_by_map = defaultdict(lambda: defaultdict(list))

########## 7. 训练主循环 ##########

# 使用命令行参数重新初始化全局规划器
global_planner = None
if args.guidance_enabled:
    global_planner = GlobalPlanner(
        resolution=args.planner_resolution,
        margin=args.planner_margin,
        device=device
    )
    print(f"[GlobalPlanner] Initialized with resolution={args.planner_resolution}m, margin={args.planner_margin}m")
else:
    print("[GlobalGuidance] Disabled; no A*/Dijkstra guidance will be computed")

current_precomputed_map_idx = int(INITIAL_PRECOMPUTED_MAP_IDX)
current_precomputed_map_type = str(INITIAL_PRECOMPUTED_MAP_TYPE) if args.use_precomputed_geometry_maps else "random"
current_precomputed_map_file = str(INITIAL_PRECOMPUTED_MAP_FILE)
current_precomputed_stage = "none" if args.use_precomputed_geometry_maps else "online_random"
current_precomputed_stage_update_count = 0
current_precomputed_active_types = ""
precomputed_type_offsets = defaultdict(int)
latest_viz_by_map_type = {map_type: None for map_type in VIZ_MAP_TYPES}
best_meta_loss = float('inf')
best_meta_loss_step = 0

terminal_log_interval = max(1, int(args.terminal_log_interval))
tb_scalar_interval = int(args.debug_scalar_interval)
pbar = tqdm(range(args.num_iters), ncols=120, miniters=terminal_log_interval)
B = args.batch_size
cycle_len = args.lgn_steps + args.worker_steps
maze_update_counter = 0
last_sparse_temporal_bucket = -1

for i in pbar:
    # This loop runs at module scope. Release graph-bearing names left by the
    # previous iteration before constructing the next rollout; otherwise
    # retained higher-order/diagnostic graphs accumulate and eventually OOM.
    release_autograd_graph_references(globals())

    term_log_now = ((i + 1) % terminal_log_interval == 0)
    artifact_save_now = is_artifact_save_iter(i, args)
    tb_log_now = (
        tb_scalar_interval > 0 and (
            i == 0
            or ((i + 1) % tb_scalar_interval == 0)
            or ((i + 1) == args.num_iters)
        )
    )
    cycle_pos = i % cycle_len
    train_lgn_phase = cycle_pos < args.lgn_steps
    phase_str = f"LGN ({cycle_pos+1}/{args.lgn_steps})" if train_lgn_phase else f"Work ({cycle_pos-args.lgn_steps+1}/{args.worker_steps})"
    env.set_meta_differentiable_mode(train_lgn_phase)
    if _diag_should_log(i, args):
        print(
            f"[DIAG iter={i}] phase={phase_str}, train_lgn_phase={train_lgn_phase}, "
            f"cycle_pos={cycle_pos}, cycle_len={cycle_len}, lgn_steps={args.lgn_steps}, worker_steps={args.worker_steps}"
        )

    regenerate_map_now = False
    if args.use_precomputed_geometry_maps:
        stage_name, active_map_types = _precomputed_curriculum_stage(i)
        stage_changed = stage_name != current_precomputed_stage
        update_precomputed_map = (maze_update_counter % args.maze_update_interval == 0) or stage_changed
        if update_precomputed_map:
            if stage_changed:
                current_precomputed_stage = stage_name
                current_precomputed_stage_update_count = 0
                current_precomputed_active_types = ",".join(active_map_types)
                stage_msg = (
                    f"iter={i + 1}, stage={stage_name}, "
                    f"active_types={current_precomputed_active_types}"
                )
                print(f"[PrecomputedMapCurriculum] {stage_msg}")
                if tb_log_now or stage_changed:
                    writer.add_text("Map/Curriculum_Stage", stage_msg, i + 1)

            current_precomputed_map_idx, current_precomputed_map_type = _select_precomputed_curriculum_map(
                active_types=active_map_types,
                stage_update_count=current_precomputed_stage_update_count,
                type_offsets=precomputed_type_offsets,
                type_indices=PRECOMPUTED_MAP_TYPE_INDICES,
            )
            current_precomputed_stage_update_count += 1
            current_precomputed_map_file = os.path.basename(POTENTIAL_MAP_CACHE.map_files[current_precomputed_map_idx])
            env.current_map_idx = current_precomputed_map_idx
            map_data_cur = POTENTIAL_MAP_CACHE.get_map(current_precomputed_map_idx)
            _align_env_goal_planes_to_precomputed_map(map_data_cur, env, map_idx_hint=current_precomputed_map_idx)
            env.reset_from_precomputed_map(map_data_cur)
            map_msg = (
                f"iter={i + 1}, idx={current_precomputed_map_idx}, "
                f"type={current_precomputed_map_type}, file={current_precomputed_map_file}"
            )
            if tb_log_now or stage_changed:
                _resolve_tb_writer(current_precomputed_map_type, writer, map_writers).add_text(
                    "Map/Current_Precomputed_File",
                    map_msg,
                    i + 1,
                )
            if term_log_now or stage_changed:
                print(f"[PrecomputedMapCurriculum] {map_msg}")
        else:
            env.reset_drone_only()
    else:
        regenerate_map_now = maze_update_counter % args.maze_update_interval == 0
        if regenerate_map_now:
            env.reset()  # New independent obstacle coordinates and sampled density.
            current_precomputed_map_idx += 1
            env.current_map_idx = current_precomputed_map_idx
            if term_log_now or i == 0:
                print(
                    f"[OnlineMap] iter={i + 1}, random_map_idx={current_precomputed_map_idx}, "
                    f"easy_density={env._effective_reference_density('easy'):.2f}x, "
                    f"hard_density={env._effective_reference_density('hard'):.2f}x"
                )
        else:
            env.reset_drone_only()  # keep maze, reset drones only
    maze_update_counter += 1
    worknet.reset()

    # Match the official training distribution: one command is sampled per
    # rollout (and per multi-drone group), then held fixed for every timestep.
    command_speed = sample_command_speed(
        B,
        args.command_speed_min,
        args.command_speed_max,
        args.speed_mtp,
        device=device,
        dtype=env.p.dtype,
        n_drones_per_group=getattr(env, 'n_drones_per_group', 1),
        randomize=args.random_command_speed,
        fixed_speed=args.command_speed,
    )

    p_history, p_next_history, v_history, a_history, vec_to_pt_history = [], [], [], [], []
    rpy_history = []
    R_history = []  # 记录姿态矩阵用于可视化
    real_act_history = []
    target_v_history = []
    v_pred_history = []
    depth_history = []
    act_buffer = [env.act.detach()] * 2
    trajectory_lgn_weights = []
    yaw_rate_cmd_history = []
    act_for_diag = None
    dist_obj_history = []
    actual_dist_obj_history = []
    geom_feat_last = None
    progress_feat_last = None

    h = None
    lgn_hx = None
    # Visualization artifacts are written only on the unified artifact cadence.
    # Do not collect rollout frames on ordinary training iterations.
    capture_viz_now = (
        artifact_save_now
        and current_precomputed_map_type in VIZ_MAP_TYPES
    )
    rollout_steps = args.lgn_timesteps if train_lgn_phase else args.timesteps

    ###### A. Rollout ######
    for t in range(rollout_steps):
        ctl_dt = 1.0 / 15.0
        depth, flow = env.render(ctl_dt)
        depth = sanitize_tensor(depth, nan=24.0, posinf=24.0, neginf=0.3)

        if capture_viz_now:
            # Keep frames on GPU during rollout.  A single batched CPU transfer
            # is performed after rollout instead of synchronizing every step.
            depth_history.append(depth[0].detach())

        p_history.append(env.p)
        v_history.append(env.v)
        a_history.append(env.a)
        vec_curr_samples = env.find_vec_to_nearest_pt()
        vec_curr, dist_obj_curr = select_min_clearance_obstacle_sample(
            vec_curr_samples, env.margin, sample_dim=0,
        )
        # Sample 0 uses tau=0, so it represents the current physical position.
        # Keep it separate from look-ahead clearance for evaluation metrics.
        actual_dist_obj_curr = safe_l2_norm(vec_curr_samples[0], dim=-1) - env.margin
        vec_to_pt_history.append(vec_curr)
        dist_obj_history.append(dist_obj_curr)
        actual_dist_obj_history.append(actual_dist_obj_curr)
        rpy_history.append(rotation_matrix_to_rpy_deg(env.R).detach())
        R_history.append(env.R.detach().clone())  # 保存姿态矩阵

        target_v_raw_curr = env.p_target - env.p.detach()
        target_v, _ = build_command_velocity(
            env.p.detach(), env.p_target, command_speed, args.goal_slowdown_time
        )
        target_v_history.append(target_v)

        R = build_yaw_frame(env.R) if use_attitude_v2 else env.R
        state_list = [torch.squeeze(target_v[:, None] @ R, 1), env.R[:, 2], env.margin[:, None]]
        if use_attitude_v2:
            heading_ref_world, heading_ref_local_xy, yaw_error = compute_heading_reference(env, R)
            yaw_rate_norm = getattr(env, "yaw_rate", torch.zeros((B, 1), device=device)) / float(yaw_rate_max)
            state_list.extend([heading_ref_local_xy, yaw_rate_norm])
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom: state_list.insert(0, local_v)
        
        state_tensor = sanitize_tensor(
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
            p_history_list=p_history,
            v_history_list=v_history,
            dist_obj_history_list=dist_obj_history,
            p_target=env.p_target,
            window=8,
        )
        context_feat_raw = extract_worker_context_features(
            p_history_list=p_history,
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
        geom_feat_last = geom_feat
        progress_feat_last = progress_feat

        # LGN forward:
        # - LGN phase: keep graph for unrolled second-order path.
        # - Worker phase: freeze LGN graph to avoid unnecessary memory overhead.
        if train_lgn_phase:
            current_weights, lgn_hx = lgn(
                x_pooled,
                state_tensor,
                geom_feat,
                progress_feat,
                lgn_hx,
            )
        else:
            with torch.no_grad():
                current_weights, lgn_hx = lgn(
                    x_pooled,
                    state_tensor,
                    geom_feat,
                    progress_feat,
                    lgn_hx,
                )

        if t == 0 and _diag_should_log(i, args):
            first_lgn_weight = current_weights[0, 0] if current_weights.numel() > 0 else None
            print(
                f"[DIAG iter={i} t=0] current_weights={_diag_grad_meta(current_weights)}, "
                f"lgn_hx={_diag_grad_meta(lgn_hx)}, "
                f"first_lgn_weight={_diag_grad_meta(first_lgn_weight)}"
            )
            expected = "requires_grad=True" if train_lgn_phase else "requires_grad=False"
            actual = bool(current_weights.requires_grad)
            print(
                f"[DIAG iter={i} t=0] LGN mode expectation: {expected}, "
                f"actual_requires_grad={actual}"
            )
            if train_lgn_phase and not actual:
                print(
                    f"[DIAG iter={i} t=0][ALERT] LGN phase but current_weights.requires_grad=False; "
                    "possible second-order path break."
                )
        trajectory_lgn_weights.append(current_weights)

        # Worker Forward: consume state + geometry + progress features.
        worker_input = torch.cat([state_tensor, geom_feat, progress_feat], dim=-1)
        act, _, h = worknet(x_pooled, worker_input, h)
        act = sanitize_tensor(act, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        act_for_diag = act
        a_pred, v_pred, yaw_rate_cmd = decode_worker_action(act, R, yaw_rate_max)
        v_pred_history.append(v_pred)
        real_act = (
            (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None]
            + env.g_std
        )
        real_act = sanitize_tensor(real_act, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        real_act_history.append(real_act.detach())
        act_buffer.append(real_act)

        if use_attitude_v2:
            heading_v_ref = env.v

            heading_ref_world, _, _, yaw_rate_rule, _ = \
                compute_velocity_heading_command(
                    R_yaw=R,
                    v_ref_world=heading_v_ref,
                    yaw_rate_max_value=yaw_rate_max,
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
                -float(yaw_rate_max),
                float(yaw_rate_max),
            )

            yaw_rate_cmd_history.append(yaw_rate_cmd_final)
            env.run(
                real_act,
                ctl_dt,
                heading_ref=heading_ref_world,
                yaw_rate_cmd=yaw_rate_cmd_final,
                yaw_rate_max=yaw_rate_max,
            )
        else:
            env.run(real_act, ctl_dt, target_v_raw_curr)

        # Pair the LGN weight and action at step t with the state transition
        # produced by that same action, including the final rollout action.
        p_next_history.append(env.p)

        # Keep full horizon so in-goal staying can continuously accumulate arrival reward.

        if args.detach_interval > 0 and (t + 1) % args.detach_interval == 0:
            if h is not None:
                h = h.detach()
            # LGN phase 时保留 lgn_hx 梯度，Worker phase 时截断
            if lgn_hx is not None and not train_lgn_phase:
                lgn_hx = lgn_hx.detach()

    ###### B. Loss Calculation (Step-wise) ######
    p_history = torch.stack(p_history)     # [T, B, 3]
    p_next_history = torch.stack(p_next_history)  # [T, B, 3], post-action states
    v_history = torch.stack(v_history)     # [T, B, 3]
    target_v_history = torch.stack(target_v_history)  # [T, B, 3]
    v_pred_history = torch.stack(v_pred_history)  # [T, B, 3]
    a_history = torch.stack(a_history)     # [T, B, 3]
    act_buffer = torch.stack(act_buffer)   # [T+2, B, 3]
    weights_seq = torch.stack(trajectory_lgn_weights)  # [T, B, 6]
    if weights_seq.shape[-1] != 6:
        raise RuntimeError(
            "LGN must output exactly 6 weights: Avoidance, Exploration, Turn, "
            "Progress, Smoothness, Energy; "
            f"got shape {tuple(weights_seq.shape)}"
        )
    if _diag_should_log(i, args):
        print(f"[DIAG iter={i}] weights_seq: {_diag_grad_meta(weights_seq)}")
        if train_lgn_phase and not weights_seq.requires_grad:
            print(
                f"[DIAG iter={i}][ALERT] LGN phase but weights_seq.requires_grad=False; "
                "possible second-order path break."
            )
    rpy_history = torch.stack(rpy_history) # [T, B, 3]
    R_history = torch.stack(R_history)     # [T, B, 3, 3]
    real_act_history = torch.stack(real_act_history) # [T, B, 3]

    vec_to_pt = torch.stack(vec_to_pt_history)
    
    # 1. 计算各项 Raw Loss (保留 [T, B] 维度用于 Step-wise 加权)

    # 碰撞距离。
    # Each step already contains the most dangerous of the 10 look-ahead samples.
    dist_obj = torch.stack(dist_obj_history)  # [T, B]
    actual_dist_obj = torch.stack(actual_dist_obj_history)  # [T, B], tau=0 only

    dist_to_goal = safe_l2_norm(env.p_target - p_history, dim=-1)  # [T, B]

    with torch.no_grad():
        v_to_pt = torch.ones_like(dist_obj)
        if dist_obj.shape[0] > 1:
            v_to_pt[1:] = (-torch.diff(dist_obj, 1, 0) * 135.0).clamp_min(1.0)

    collision_depth = F.relu(-dist_obj)
    loss_avoidance_seq = v_to_pt * (1.0 - dist_obj).relu().pow(2)
    loss_collision_seq = F.softplus(dist_obj.mul(-32.0)) * v_to_pt

    # 注意: compute_overlap_loss_per_step 返回 [B, T], 需要 permute 成 [T, B]
    loss_exploration_seq = compute_overlap_loss_per_step(
        p_history, sigma=1.0, time_window=int(args.exploration_time_window)
    ).permute(1, 0)
    loss_turn_base_seq = compute_turn_preference_loss(
        v_history,
        speed_threshold=args.turn_speed_threshold,
        speed_softness=args.turn_speed_softness,
        soft_angle_deg=args.turn_soft_angle_deg,
    )
    loss_turn_seq = loss_turn_base_seq
    progress_step_scale = max(float(env.max_speed) * float(ctl_dt), 1e-3)
    loss_progress_seq = compute_goal_progress_preference_loss(
        p_history,
        p_next_history,
        env.p_target,
        step_scale=progress_step_scale,
    )
    loss_smoothness_seq, loss_jerk_seq, loss_snap_seq = \
        compute_action_smoothness_losses_per_step(
            act_buffer,
            env.g_std,
            control_hz=1.0 / float(ctl_dt),
            jerk_weight=args.meta_smooth_jerk_weight,
            snap_weight=args.meta_smooth_snap_weight,
        )
    loss_energy_seq = compute_action_energy_loss_per_step(
        act_buffer[2:],
        action_scale=args.energy_action_scale,
    )
    if use_attitude_v2 and len(yaw_rate_cmd_history) > 0:
        yaw_rate_cmd_seq = torch.stack(yaw_rate_cmd_history)  # [T, B, 1]
        if yaw_rate_cmd_seq.shape[0] > 1:
            loss_yaw_smooth = yaw_rate_cmd_seq.diff(1, 0).pow(2).mean()
        else:
            loss_yaw_smooth = torch.tensor(0.0, device=device, dtype=v_history.dtype)
    else:
        loss_yaw_smooth = torch.tensor(0.0, device=device, dtype=v_history.dtype)

    # Fixed task supervision, independent of the six LGN-generated weights.
    # The actual-velocity term enforces the sampled command; the auxiliary
    # prediction term restores the official Worker's velocity head.
    loss_velocity_track = compute_velocity_tracking_loss(
        v_history,
        target_v_history,
        window=args.velocity_track_window,
    )
    loss_velocity_predict = F.mse_loss(v_pred_history, v_history.detach())

    loss_stuck_seq, loss_collision_duration_seq, stuck_ratio = compute_stuck_loss(
        p_history, collision_depth,
        stuck_window=args.stuck_window,
        displacement_threshold=args.stuck_displacement_threshold,
    )
    actual_T = p_history.shape[0]

    # 只保留越界安全惩罚；不再将整段轨迹拉向固定高度。
    z_pos = p_history[:, :, 2]  # [T, B]
    z_min, z_max = 0.0, 5.0
    loss_height_bounds_seq = (
        F.softplus((z_pos - z_max) * 20.0)
        + F.softplus((z_min - z_pos) * 20.0)
    )

    # LGN 输出的六个动态损失权重均由输出头约束为非负。
    weights_seq_raw = weights_seq
    if _diag_should_log(i, args):
        print(f"[DIAG iter={i}] weights_seq_raw: requires_grad={weights_seq_raw.requires_grad}, grad_fn={type(weights_seq_raw.grad_fn).__name__ if weights_seq_raw.grad_fn else 'None'}")
        print(
            f"[DIAG iter={i}] loss_raw requires_grad: avoid={loss_avoidance_seq.requires_grad}, "
            f"expl={loss_exploration_seq.requires_grad}, "
            f"turn={loss_turn_seq.requires_grad}, progress={loss_progress_seq.requires_grad}, "
            f"smooth={loss_smoothness_seq.requires_grad}, energy={loss_energy_seq.requires_grad}"
        )
    # 保留 raw/effective 别名，便于诊断和记录两处使用统一张量。
    effective_weights_seq = weights_seq_raw
    if _diag_should_log(i, args):
        print(
            f"[DIAG iter={i}] effective_weights: requires_grad={effective_weights_seq.requires_grad}, "
            f"grad_fn={type(effective_weights_seq.grad_fn).__name__ if effective_weights_seq.grad_fn else 'None'}"
        )

    # 2. Step-wise 加权 (Broadcasting: [T, B] * [T, B])
    weighted_loss_map = (
        effective_weights_seq[:, :, 0] * loss_avoidance_seq +
        effective_weights_seq[:, :, 1] * loss_exploration_seq +
        effective_weights_seq[:, :, 2] * loss_turn_seq +
        effective_weights_seq[:, :, 3] * loss_progress_seq +
        effective_weights_seq[:, :, 4] * loss_smoothness_seq +
        effective_weights_seq[:, :, 5] * loss_energy_seq
    )

    # 3. 最终 Proxy Loss
    proxy_weighted_core = weighted_loss_map.mean()
    proxy_loss = proxy_weighted_core
    if _diag_should_log(i, args):
        print(
            f"[DIAG iter={i}] weighted_loss_map: requires_grad={weighted_loss_map.requires_grad}, "
            f"grad_fn={type(weighted_loss_map.grad_fn).__name__ if weighted_loss_map.grad_fn else 'None'}"
        )
        print(
            f"[DIAG iter={i}] proxy_loss: requires_grad={proxy_loss.requires_grad}, "
            f"grad_fn={type(proxy_loss.grad_fn).__name__ if proxy_loss.grad_fn else 'None'}"
        )
        _diag_tensor_finite("act_for_diag", act_for_diag, i)
        _diag_tensor_finite("real_act_history", real_act_history, i)
        _diag_tensor_finite("p_history", p_history, i)
        _diag_tensor_finite("v_history", v_history, i)
        _diag_tensor_finite("a_history", a_history, i)
        _diag_tensor_finite("vec_to_pt", vec_to_pt, i)
        _diag_tensor_finite("dist_obj", dist_obj, i)
        _diag_tensor_finite("loss_avoidance_seq", loss_avoidance_seq, i)
        _diag_tensor_finite("loss_exploration_seq", loss_exploration_seq, i)
        _diag_tensor_finite("loss_turn_seq", loss_turn_seq, i)
        _diag_tensor_finite("loss_progress_seq", loss_progress_seq, i)
        _diag_tensor_finite("loss_smoothness_seq", loss_smoothness_seq, i)
        _diag_tensor_finite("loss_energy_seq", loss_energy_seq, i)
        _diag_tensor_finite("loss_yaw_smooth", loss_yaw_smooth, i)
        _diag_tensor_finite("weights_seq_raw", weights_seq_raw, i)
        _diag_tensor_finite("weighted_loss_map", weighted_loss_map, i)
        _diag_tensor_finite("proxy_loss", proxy_loss, i)

    # Direct Worker task objective. Arrival, command-speed compliance,
    # velocity prediction, and collision safety all remain fixed-weight terms
    # that the LGN cannot suppress.
    worker_terminal_loss = safe_l2_norm(p_history[-1] - env.p_target, dim=-1).mean()
    worker_arrival_reward, _, _ = compute_arrival_reward(
        p_history,
        env.p_target,
        radius=args.meta_arrival_reward_radius,
        temperature=args.arrival_soft_temperature,
        time_step=1.0 / 15.0,
    )
    worker_arrival_task_loss = (
        worker_terminal_loss
        - args.meta_arrival_reward_weight * worker_arrival_reward
    )
    worker_collision_loss = loss_collision_seq.mean()
    worker_primary_task_loss = (
        float(args.worker_arrival_weight) * worker_arrival_task_loss
        + float(args.worker_velocity_track_weight) * loss_velocity_track
        + float(args.worker_velocity_predict_weight) * loss_velocity_predict
        + float(args.worker_collision_weight) * worker_collision_loss
    )

    # --- Meta Loss Components ---
    # Worker phase only needs the direct arrival objective above; detach the
    # remaining meta diagnostics to avoid retaining unrelated graph branches.
    if train_lgn_phase:
        meta_p_history = p_history
        meta_v_history = v_history
        meta_a_history = a_history
        meta_act_buffer = act_buffer
        meta_loss_collision_seq = loss_collision_seq
        meta_loss_stuck_seq = loss_stuck_seq
        meta_loss_height_bounds_seq = loss_height_bounds_seq
        meta_loss_progress_seq = loss_progress_seq
        meta_loss_jerk_seq = loss_jerk_seq
        meta_loss_snap_seq = loss_snap_seq
        meta_loss_energy_seq = loss_energy_seq
        meta_loss_velocity_track = loss_velocity_track
    else:
        meta_p_history = p_history.detach()
        meta_v_history = v_history.detach()
        meta_a_history = a_history.detach()
        meta_act_buffer = act_buffer.detach()
        meta_loss_collision_seq = loss_collision_seq.detach()
        meta_loss_stuck_seq = loss_stuck_seq.detach()
        meta_loss_height_bounds_seq = loss_height_bounds_seq.detach()
        meta_loss_progress_seq = loss_progress_seq.detach()
        meta_loss_jerk_seq = loss_jerk_seq.detach()
        meta_loss_snap_seq = loss_snap_seq.detach()
        meta_loss_energy_seq = loss_energy_seq.detach()
        meta_loss_velocity_track = loss_velocity_track.detach()

    # Final distance remains useful for monitoring, but it is not an LGN/meta
    # optimization term and must not retain an unnecessary gradient branch.
    with torch.no_grad():
        loss_meta_pos = safe_l2_norm(meta_p_history[-1] - env.p_target, dim=-1).mean()
    loss_meta_arrival_reward, meta_arrival_hit_rate, meta_arrival_best_dist = compute_arrival_reward(
        meta_p_history,
        env.p_target,
        radius=args.meta_arrival_reward_radius,
        temperature=args.arrival_soft_temperature,
        time_step=1.0 / 15.0,
    )
    loss_meta_coll = meta_loss_collision_seq.mean()
    loss_meta_ctrl = safe_l2_norm(meta_act_buffer, dim=-1).sum()
    loss_meta_progress = meta_loss_progress_seq.mean()
    loss_meta_jerk = meta_loss_jerk_seq.mean()
    loss_meta_snap = meta_loss_snap_seq.mean()
    loss_meta_energy = meta_loss_energy_seq.mean()
    loss_meta_height_bounds = meta_loss_height_bounds_seq.mean()
    # --- 全局规划引导损失 ---
    guidance_active_this_phase = bool(
        args.guidance_enabled and (train_lgn_phase or args.guidance_all_phases)
    )
    if guidance_active_this_phase:
        if train_lgn_phase:
            loss_meta_guidance, guidance_components = compute_global_guidance_meta_loss(
                env, meta_p_history, meta_v_history, env.p_target, vec_to_pt, dist_obj,
                config=args,
                potential_map_cache=POTENTIAL_MAP_CACHE,
                planner=global_planner,
                a_history=meta_a_history,
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
            with torch.no_grad():
                loss_meta_guidance, guidance_components = compute_global_guidance_meta_loss(
                    env,
                    meta_p_history.detach(),
                    meta_v_history.detach(),
                    env.p_target,
                    vec_to_pt.detach(),
                    dist_obj.detach(),
                    config=args,
                    potential_map_cache=POTENTIAL_MAP_CACHE,
                    planner=global_planner,
                    a_history=meta_a_history.detach(),
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
            loss_meta_guidance = loss_meta_guidance.detach()
    else:
        zero = torch.tensor(0.0, device=p_history.device)
        loss_meta_guidance = zero
        guidance_components = {
            'dir_align': zero,
            'speed_diff': zero,
            'overspeed': zero,
            'underspeed': zero,
            'lateral_error': zero,
            'accel_mismatch': zero,
            'escape': zero,
            'depth': zero,
            'recovery_speed': zero,
            'valid_ratio': zero,
            'invalid_ratio': zero,
            'collision_ratio': zero,
            'sample_count': 0.0,
            'avg_curvature': 0.0,
            'avg_path_progress': 0.0,
            'avg_lateral_error': 0.0,
            'max_lateral_error': 0.0,
            'planner_success_ratio': 0.0,
            'avg_ref_speed': 0.0,
            'sampled_astar_paths': [],
            'field_dir_align': 0.0,
        }

    # 训练目标使用碰撞/高度越界安全项/引导/进展/平滑/能耗；最终位置距离仅用于日志监控。
    loss_meta_stuck = meta_loss_stuck_seq.mean()
    meta_loss = (
        loss_meta_coll +
        loss_meta_height_bounds +
        args.meta_guidance_weight * loss_meta_guidance +
        args.meta_smooth_jerk_weight * loss_meta_jerk +
        args.meta_smooth_snap_weight * loss_meta_snap +
        args.meta_progress_weight * loss_meta_progress +
        args.meta_energy_weight * loss_meta_energy +
        args.meta_velocity_track_weight * meta_loss_velocity_track +
        args.stuck_loss_weight * loss_meta_stuck +
        args.coef_yaw_smooth * loss_yaw_smooth
        - args.meta_arrival_reward_weight * loss_meta_arrival_reward
    )
    worker_proxy_loss = float(args.worker_proxy_weight) * proxy_loss
    worker_total_loss = worker_proxy_loss + worker_primary_task_loss
    if _diag_should_log(i, args):
        _diag_tensor_finite("loss_meta_pos", loss_meta_pos, i)
        _diag_tensor_finite("loss_meta_arrival_reward", loss_meta_arrival_reward, i)
        _diag_tensor_finite("loss_meta_coll", loss_meta_coll, i)
        _diag_tensor_finite("loss_meta_height_bounds", loss_meta_height_bounds, i)
        _diag_tensor_finite("loss_meta_guidance", loss_meta_guidance, i)
        _diag_tensor_finite("loss_meta_jerk", loss_meta_jerk, i)
        _diag_tensor_finite("loss_meta_snap", loss_meta_snap, i)
        _diag_tensor_finite("loss_meta_progress", loss_meta_progress, i)
        _diag_tensor_finite("loss_meta_energy", loss_meta_energy, i)
        _diag_tensor_finite("loss_meta_stuck", loss_meta_stuck, i)
        _diag_tensor_finite("loss_velocity_track", loss_velocity_track, i)
        _diag_tensor_finite("loss_velocity_predict", loss_velocity_predict, i)
        _diag_tensor_finite("loss_yaw_smooth", loss_yaw_smooth, i)
        _diag_tensor_finite("meta_loss", meta_loss, i)
        _diag_tensor_finite("worker_primary_task_loss", worker_primary_task_loss, i)
        _diag_tensor_finite("worker_total_loss", worker_total_loss, i)
    # Keep root losses untouched for higher-order gradients; skip iteration via finite checks below.

    ###### C. Optimization ######
    optim_worker.zero_grad()
    optim_lgn.zero_grad()
    lgn_update_loss = 0.0
    worker_grad_norm = 0.0
    worker_grad_max = 0.0
    worker_grad_nonfinite = 0.0
    worker_grad_elems = 0.0
    worker_clip_pre = 0.0
    worker_proxy_grad_norm = 0.0
    worker_arrival_grad_norm = 0.0
    worker_proxy_grad_scale = 0.0
    worker_proxy_arrival_grad_cosine = 0.0
    worker_proxy_conflict_projected = 0.0
    lgn_grad_norm = 0.0
    lgn_grad_max = 0.0
    lgn_grad_nonfinite = 0.0
    lgn_grad_elems = 0.0
    lgn_clip_pre = 0.0
    lgn_meta_probe_norm = 0.0
    lgn_meta_probe_nonfinite = 0.0
    lgn_meta_probe_elems = 0.0
    gradient_alignment_stats = None
    sparse_temporal_stats = None
    gradient_alignment_loss = torch.tensor(0.0, device=device)
    gradient_alignment_cosine = torch.tensor(0.0, device=device)
    gradient_alignment_usable = False
    meta_pos_ur = torch.tensor(0.0, device=device)
    meta_coll_ur = torch.tensor(0.0, device=device)
    meta_ctrl_ur = torch.tensor(0.0, device=device)
    meta_arrival_reward_ur = torch.tensor(0.0, device=device)
    meta_arrival_hit_rate_ur = torch.tensor(0.0, device=device)
    meta_arrival_best_dist_ur = torch.tensor(0.0, device=device)
    meta_terms_ur = {}

    rollout_is_finite = bool(
        torch.isfinite(proxy_loss).all()
        and torch.isfinite(meta_loss).all()
        and torch.isfinite(worker_total_loss).all()
        and torch.isfinite(weights_seq).all()
        and torch.isfinite(p_history).all()
        and torch.isfinite(v_history).all()
    )

    if not rollout_is_finite:
        if term_log_now:
            pbar.set_description(f"[{phase_str}] non-finite rollout skipped")
        continue

    if train_lgn_phase:
        # ===== Unrolled Bilevel: 可微内循环 =====
        # Step 1: use the same proxy + direct task objective as the persistent
        # Worker update, while retaining the graph for bilevel differentiation.
        fast_params = dict(worknet.named_parameters())
        inner_update_is_finite = True

        for _inner in range(args.inner_steps):
            inner_grads = torch.autograd.grad(
                worker_total_loss, tuple(fast_params.values()),
                create_graph=True, allow_unused=True, retain_graph=True,
            )

            inner_sq_terms = [(g * g).sum() for g in inner_grads if g is not None]
            if inner_sq_terms:
                inner_grad_norm_tensor = torch.sqrt(torch.stack(inner_sq_terms).sum() + 1e-12)
                if not bool(torch.isfinite(inner_grad_norm_tensor).item()):
                    inner_update_is_finite = False
                    break
                inner_scale = (args.grad_clip_norm / (inner_grad_norm_tensor + 1e-6)).clamp(max=1.0)
                inner_grads = tuple((g * inner_scale) if g is not None else None for g in inner_grads)

            if _inner == 0 and _diag_should_log(i, args):
                fast_param_values = tuple(fast_params.values())

                if args.diag_second_order:
                    # Probe weighted per-term proxy components so each branch truly depends on LGN weights.
                    weighted_avoid = (effective_weights_seq[:, :, 0] * loss_avoidance_seq).mean()
                    weighted_expl = (effective_weights_seq[:, :, 1] * loss_exploration_seq).mean()
                    weighted_turn = (effective_weights_seq[:, :, 2] * loss_turn_seq).mean()
                    weighted_progress = (effective_weights_seq[:, :, 3] * loss_progress_seq).mean()
                    weighted_smooth = (effective_weights_seq[:, :, 4] * loss_smoothness_seq).mean()
                    weighted_energy = (effective_weights_seq[:, :, 5] * loss_energy_seq).mean()

                    g_avoid = _grad_or_none_tuple(weighted_avoid, fast_param_values)
                    g_expl = _grad_or_none_tuple(weighted_expl, fast_param_values)
                    g_turn = _grad_or_none_tuple(weighted_turn, fast_param_values)
                    g_progress = _grad_or_none_tuple(weighted_progress, fast_param_values)
                    g_smooth = _grad_or_none_tuple(weighted_smooth, fast_param_values)
                    g_energy = _grad_or_none_tuple(weighted_energy, fast_param_values)

                    lgn_param_list = list(lgn.parameters())
                    _diag_grad_tuple_to_params("avoidance(weighted) second_order(worker_grad)->lgn", g_avoid, lgn_param_list, i)
                    _diag_grad_tuple_to_params("exploration(weighted) second_order(worker_grad)->lgn", g_expl, lgn_param_list, i)
                    _diag_grad_tuple_to_params("turn(weighted) second_order(worker_grad)->lgn", g_turn, lgn_param_list, i)
                    _diag_grad_tuple_to_params("progress(weighted) second_order(worker_grad)->lgn", g_progress, lgn_param_list, i)
                    _diag_grad_tuple_to_params("smoothness(weighted) second_order(worker_grad)->lgn", g_smooth, lgn_param_list, i)
                    _diag_grad_tuple_to_params("energy(weighted) second_order(worker_grad)->lgn", g_energy, lgn_param_list, i)
                    _diag_grad_tuple_to_params("worker_total second_order(worker_grad)->lgn", inner_grads, lgn_param_list, i)

                    _diag_grad_tuple_to_params("inner_grads(sum) -> lgn", inner_grads, lgn_param_list, i)
                    toy_grad = tuple((g.pow(2) if g is not None else None) for g in inner_grads)
                    _diag_grad_tuple_to_params("toy_grad(sqsum) -> lgn", toy_grad, lgn_param_list, i)

                inner_norm, inner_nonfinite, inner_elems = get_grad_norm_from_grads(inner_grads)
                print(f"[DIAG iter={i}] inner_grads finite: nonfinite={inner_nonfinite}/{inner_elems}")

            fast_params = {
                name: (p - args.inner_lr * g
                       if g is not None else p)
                for (name, p), g in zip(fast_params.items(), inner_grads)
            }

        if not inner_update_is_finite:
            if term_log_now:
                pbar.set_description(f"[{phase_str}] non-finite inner-update skipped")
        else:
            if _diag_should_log(i, args):
                fast_param_vals = list(fast_params.values())
                if len(fast_param_vals) > 0:
                    _diag_tensor_finite("first_fast_param", fast_param_vals[0], i)
                    _diag_output_to_params_count("fast_params(sum) -> lgn", sum(fp.sum() for fp in fast_param_vals), lgn.parameters(), i)

            # Step 2: 用虚拟更新后的 worker 做验证 rollout → meta_loss
            (meta_loss_unrolled, meta_pos_ur, meta_coll_ur, meta_ctrl_ur,
             meta_arrival_reward_ur, meta_arrival_hit_rate_ur, meta_arrival_best_dist_ur,
             meta_terms_ur) = \
                unrolled_meta_rollout(
                    env,
                    worknet,
                    fast_params,
                    args,
                    B,
                    device,
                    POTENTIAL_MAP_CACHE,
                    global_planner,
                    command_speed,
                    iter_idx=i,
                )
            if not torch.isfinite(meta_loss_unrolled):
                if term_log_now:
                    pbar.set_description(f"[{phase_str}] non-finite unroll skipped")
            else:
                # LGN auxiliary objective: align only the Worker gradient of the
                # LGN-weighted proxy core with the unrolled meta gradient. Direct
                # arrival, yaw smoothing, and all other fixed-weight terms are
                # intentionally excluded from the proxy side.
                proxy_alignment_grads = torch.autograd.grad(
                    proxy_weighted_core,
                    tuple(worknet.parameters()),
                    allow_unused=True,
                    retain_graph=True,
                    create_graph=True,
                )
                fast_param_values = tuple(fast_params.values())
                lgn_param_values = tuple(lgn.parameters())
                diag_grad_now = _diag_should_log(i, args)
                sparse_temporal_interval = int(args.sparse_temporal_monitor_interval)
                sparse_temporal_bucket = (
                    i // sparse_temporal_interval
                    if sparse_temporal_interval > 0 else 0
                )
                sparse_temporal_monitor_now = (
                    sparse_temporal_interval > 0
                    and sparse_temporal_bucket > last_sparse_temporal_bucket
                )
                if sparse_temporal_monitor_now:
                    last_sparse_temporal_bucket = sparse_temporal_bucket
                    sparse_meta_term = meta_terms_ur.get(
                        "first_arrival_log_time_reward_weighted"
                    )
                    sparse_temporal_stats = {
                        'SparseInfluence/0_Arrival_Hit_Rate': float(
                            meta_arrival_hit_rate_ur.detach().item()
                        ),
                        'SparseInfluence/Temporal/Probe_Valid': 0.0,
                    }
                    if sparse_meta_term is not None:
                        sparse_temporal_stats[
                            'SparseInfluence/1_Sparse_Meta_Term'
                        ] = float(sparse_meta_term.detach().item())
                    if (
                        sparse_meta_term is not None
                        and sparse_meta_term.requires_grad
                    ):
                        sparse_fast_grads = torch.autograd.grad(
                            sparse_meta_term,
                            fast_param_values,
                            allow_unused=True,
                            retain_graph=True,
                            create_graph=False,
                        )
                        sparse_weight_sensitivity = \
                            compute_meta_hypergrads_from_fast_grads(
                                fast_param_values,
                                sparse_fast_grads,
                                (effective_weights_seq,),
                                retain_graph=True,
                            )[0]
                        if sparse_weight_sensitivity is not None:
                            temporal_summary = summarize_temporal_weight_sensitivity(
                                sparse_weight_sensitivity
                            )
                            sensitivity_norm = temporal_summary['norm']
                            sparse_temporal_stats.update({
                                'SparseInfluence/8_Weight_Sensitivity_Norm': sensitivity_norm,
                                'SparseInfluence/9_Weight_Sensitivity_AbsMean': temporal_summary['abs_mean'],
                                'SparseInfluence/10_Weight_Sensitivity_Nonzero_Fraction': temporal_summary['nonzero_fraction'],
                                'SparseInfluence/11_Long_Chain_Reaches_Weights': (
                                    1.0 if sensitivity_norm > 1e-12 else 0.0
                                ),
                                'SparseInfluence/12_Log10_Weight_Sensitivity_Norm': math.log10(
                                    max(sensitivity_norm, 1e-30)
                                ),
                                'SparseInfluence/13_Weight_Sensitivity_Finite_Fraction': temporal_summary['finite_fraction'],
                                'SparseInfluence/Temporal/Early_Weight_AbsMean': temporal_summary['early_abs_mean'],
                                'SparseInfluence/Temporal/Middle_Weight_AbsMean': temporal_summary['middle_abs_mean'],
                                'SparseInfluence/Temporal/Late_Weight_AbsMean': temporal_summary['late_abs_mean'],
                                'SparseInfluence/Temporal/Early_to_Late_Ratio': temporal_summary['early_to_late_ratio'],
                                'SparseInfluence/Temporal/Late_to_Early_Decay_Factor': temporal_summary['late_to_early_decay_factor'],
                                'SparseInfluence/Temporal/Early_to_Late_Log10_Ratio': temporal_summary['early_to_late_log10_ratio'],
                                'SparseInfluence/Temporal/Ratio_Valid': temporal_summary['ratio_valid'],
                                'SparseInfluence/Temporal/Probe_Valid': 1.0,
                            })
                meta_alignment_grads = torch.autograd.grad(
                    meta_loss_unrolled,
                    fast_param_values,
                    allow_unused=True,
                    # Per-term diagnostics below still need the validation graph.
                    # Normal training reuses these returned gradients and can
                    # release that graph immediately.
                    retain_graph=diag_grad_now,
                    create_graph=False,
                )
                gradient_alignment_loss, gradient_alignment_cosine, gradient_alignment_usable = \
                    compute_gradient_alignment_loss(
                        proxy_alignment_grads,
                        meta_alignment_grads,
                        reference=proxy_weighted_core,
                    )
                if tb_log_now or _diag_should_log(i, args):
                    gradient_alignment_stats = get_gradient_alignment_stats(
                        proxy_alignment_grads,
                        meta_alignment_grads,
                    )

                # Reuse d(meta_loss)/d(fast_params) as the VJP cotangent:
                #   d(meta_loss)/d(LGN) = (d fast_params/d LGN)^T @ dM/d fast_params
                # This traverses only the differentiable inner update and avoids
                # a second backward through the validation rollout.
                alignment_backprop_needed = (
                    gradient_alignment_usable
                    and float(args.lgn_grad_alignment_weight) != 0.0
                )
                meta_hyper_grads = compute_meta_hypergrads_from_fast_grads(
                    fast_param_values,
                    meta_alignment_grads,
                    lgn_param_values,
                    retain_graph=bool(alignment_backprop_needed or diag_grad_now),
                )
                lgn_meta_probe_norm, lgn_meta_probe_nonfinite, lgn_meta_probe_elems = \
                    get_grad_norm_from_grads(meta_hyper_grads)
                meta_grad_usable = (
                    lgn_meta_probe_elems > 0
                    and lgn_meta_probe_nonfinite == 0
                    and lgn_meta_probe_norm > 0.0
                    and math.isfinite(lgn_meta_probe_norm)
                )

                if diag_grad_now:
                    print(
                        f"[DIAG iter={i}] meta_loss_unrolled: requires_grad={meta_loss_unrolled.requires_grad}, "
                        f"grad_fn={type(meta_loss_unrolled.grad_fn).__name__ if meta_loss_unrolled.grad_fn else 'None'}"
                    )
                    _diag_tensor_finite("meta_loss_unrolled", meta_loss_unrolled, i)
                    print(
                        f"[DIAG iter={i}] meta_loss_unrolled -> lgn(reused-vjp): "
                        f"Norm={lgn_meta_probe_norm:.6f}, "
                        f"NonFinite={int(lgn_meta_probe_nonfinite)}/{int(lgn_meta_probe_elems)}"
                    )
                    for meta_term_name, meta_term_value in meta_terms_ur.items():
                        _diag_output_to_params(
                            f"meta_term/{meta_term_name} -> fast_params",
                            meta_term_value,
                            fast_params.values(),
                            i,
                        )
                    _diag_output_to_params("proxy_loss -> lgn", proxy_loss, lgn.parameters(), i)
                    _diag_output_to_params("meta_loss -> lgn", meta_loss, lgn.parameters(), i)

                # Step 3: use the reused pure-meta hypergradients for both the
                # usability check and the actual LGN update.  Alignment remains
                # an auxiliary gradient and is added explicitly below.
                if not meta_grad_usable:
                    if _diag_should_log(i, args):
                        print(
                            f"[DIAG iter={i}] skip LGN step: unusable meta gradients "
                            f"(meta_probe_norm={lgn_meta_probe_norm:.6f}, "
                            f"meta_probe_nonfinite={int(lgn_meta_probe_nonfinite)}/{int(lgn_meta_probe_elems)})"
                        )
                    if term_log_now:
                        pbar.set_description(f"[{phase_str}] meta-grad unusable, LGN step skipped")
                else:
                    if alignment_backprop_needed:
                        alignment_grads = torch.autograd.grad(
                            float(args.lgn_grad_alignment_weight) * gradient_alignment_loss,
                            lgn_param_values,
                            allow_unused=True,
                            retain_graph=False,
                            create_graph=False,
                        )
                    else:
                        alignment_grads = tuple(None for _ in lgn_param_values)

                    # autograd.grad does not populate parameter.grad.  Assign the
                    # exact sum dM/dphi + lambda*dL_align/dphi for AdamW.
                    for param, meta_grad, alignment_grad in zip(
                        lgn_param_values,
                        meta_hyper_grads,
                        alignment_grads,
                    ):
                        if meta_grad is None and alignment_grad is None:
                            param.grad = None
                            continue
                        total_grad = torch.zeros_like(param)
                        if meta_grad is not None:
                            total_grad = total_grad + meta_grad.detach()
                        if alignment_grad is not None:
                            total_grad = total_grad + alignment_grad.detach()
                        param.grad = total_grad

                    lgn_grad_norm, lgn_grad_max, lgn_grad_nonfinite, lgn_grad_elems = get_grad_stats(lgn)
                    if _diag_should_log(i, args):
                        print(
                            f"[DIAG iter={i}] lgn_grads: norm={lgn_grad_norm:.6f}, "
                            f"nonfinite={int(lgn_grad_nonfinite)}/{int(lgn_grad_elems)}, max={lgn_grad_max:.6f}"
                        )

                    lgn_clip_pre = float(nn.utils.clip_grad_norm_(lgn.parameters(), args.grad_clip_norm).item())
                    optim_lgn.step()
                    sanitize_module_(lgn, clamp_value=5.0)

                    lgn_update_loss = meta_loss_unrolled.detach()
    else:
        worker_params = tuple(worknet.parameters())
        proxy_worker_grads = torch.autograd.grad(
            worker_proxy_loss,
            worker_params,
            allow_unused=True,
            retain_graph=True,
            create_graph=False,
        )
        task_worker_grads = torch.autograd.grad(
            worker_primary_task_loss,
            worker_params,
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )
        merged_worker_grads, worker_merge_stats = merge_task_priority_gradients(
            proxy_worker_grads,
            task_worker_grads,
            max_proxy_to_task_ratio=args.worker_proxy_grad_ratio,
        )
        for param, grad in zip(worker_params, merged_worker_grads):
            param.grad = None if grad is None else grad
        worker_proxy_grad_norm = worker_merge_stats['proxy_norm']
        worker_arrival_grad_norm = worker_merge_stats['task_norm']
        worker_proxy_grad_scale = worker_merge_stats['proxy_scale']
        worker_proxy_arrival_grad_cosine = worker_merge_stats['cosine']
        worker_proxy_conflict_projected = worker_merge_stats['conflict_projected']
        worker_grad_norm, worker_grad_max, worker_grad_nonfinite, worker_grad_elems = get_grad_stats(worknet)
        worker_clip_pre = float(nn.utils.clip_grad_norm_(
            worknet.parameters(), args.worker_grad_clip_norm
        ).item())
        optim_worker.step()
        sanitize_module_(worknet, clamp_value=10.0)
        sched.step()

    ###### D. Logging & Saving (Enhanced) ######
    if term_log_now:
        if train_lgn_phase:
            pbar.set_description(f"[{phase_str}] W-Loss: {worker_total_loss:.3f} | M-Unroll: {lgn_update_loss:.3f}")
        else:
            pbar.set_description(f"[{phase_str}] W-Loss: {worker_total_loss:.3f} | M-Loss: {meta_loss:.3f}")
    
    with torch.no_grad():
        weights_raw_tb = weights_seq_raw.detach()
        weights_eff_tb = effective_weights_seq.detach()
        weights_raw_mean_tb = weights_raw_tb.mean(dim=(0, 1))
        weights_eff_mean_tb = weights_eff_tb.mean(dim=(0, 1))
        if weights_eff_tb.shape[0] > 0:
            weights_snapshot_eff_tb = weights_eff_tb[-1].mean(dim=0)
            snapshot_raw_flat_tb = weights_raw_tb[-1].reshape(-1)
        else:
            weights_snapshot_eff_tb = weights_eff_mean_tb
            snapshot_raw_flat_tb = weights_raw_tb.reshape(-1)
        # Evaluation uses only the current-position sample (tau=0).  Look-ahead
        # samples remain active for avoidance/collision training losses above.
        collision_free = torch.all(actual_dist_obj > 0, dim=0)
        reached_goal = torch.any(dist_to_goal < args.goal_radius, dim=0)
        success = collision_free & reached_goal
        final_dist_to_goal = dist_to_goal[-1]
        v_norm = v_history.norm(dim=-1)
        avg_speed = v_norm.mean()
        min_speed_threshold = float(env.max_speed) * 0.7
        act_cmd_mean = real_act_history.mean(dim=(0, 1))
        act_cmd_abs_mean = real_act_history.abs().mean(dim=(0, 1))
        act_cmd_norm_mean = real_act_history.norm(dim=-1).mean()
        exploration_window_effective = float(min(max(0, int(args.exploration_time_window)), max(0, actual_T - 2)))
        guidance_backprop_in_phase = 1.0 if (train_lgn_phase and guidance_active_this_phase) else 0.0
        map_type_code = PRECOMPUTED_MAP_TYPE_CODES.get(current_precomputed_map_type, -2)
        map_stage_code = PRECOMPUTED_CURRICULUM_STAGE_CODES.get(current_precomputed_stage, -2)

        log_data = {
            # === 主要Loss ===
            'Loss/1_Proxy_Total': proxy_loss,
            'Loss/1_1_Proxy_Weighted_Core': proxy_weighted_core,
            'Loss/2_Meta_Total': meta_loss,
            'Loss/3_Worker_Total': worker_total_loss,
            'Loss/3_0_Worker_Primary_Task': worker_primary_task_loss,
            'Loss/3_1_Worker_Arrival_Task': worker_arrival_task_loss,
            'Loss/3_2_Worker_Terminal_Distance': worker_terminal_loss,
            'Loss/3_3_Worker_Arrival_Reward': worker_arrival_reward,
            'Loss/3_4_Worker_Arrival_Weight': float(args.worker_arrival_weight),
            'Loss/3_5_Worker_Proxy_Weight': float(args.worker_proxy_weight),
            'Loss/3_6_Worker_Velocity_Track': loss_velocity_track,
            'Loss/3_7_Worker_Velocity_Predict': loss_velocity_predict,
            'Loss/3_8_Worker_Collision': worker_collision_loss,

            # === [增强] Proxy Loss 原始分项 (Average over Time & Batch) ===
            'Proxy_Comp/0_Avoidance': loss_avoidance_seq.mean(),
            'Proxy_Comp/0_1_Collision_Depth': collision_depth.mean(),#穿入墙体深度
            'Proxy_Comp/1_Exploration': loss_exploration_seq.mean(),
            'Proxy_Comp/2_Turn': loss_turn_seq.mean(),
            'Proxy_Comp/2_0_Turn_Base': loss_turn_base_seq.mean(),
            'Proxy_Comp/2_2_Yaw_Smooth': loss_yaw_smooth,
            'Proxy_Comp/3_Progress': loss_progress_seq.mean(),
            'Proxy_Comp/4_Smoothness': loss_smoothness_seq.mean(),
            'Proxy_Comp/4_1_Jerk': loss_jerk_seq.mean(),
            'Proxy_Comp/4_2_Snap': loss_snap_seq.mean(),
            'Proxy_Comp/5_Energy': loss_energy_seq.mean(),
            'Stuck/Ratio': stuck_ratio,
            'Stuck/Collision_Streak_Mean': loss_collision_duration_seq.mean(),
            'Stuck/Collision_Streak_Max': loss_collision_duration_seq.max(),
            'Yaw/Yaw_Smooth_Loss': loss_yaw_smooth,

            # === [增强] Meta Loss 分项 ===
            'Meta_Comp/1_Position': loss_meta_pos,
            'Meta_Comp/0_Arrival_Reward': loss_meta_arrival_reward,
            'Meta_Comp/0_Arrival_Term': -args.meta_arrival_reward_weight * loss_meta_arrival_reward,
            'Meta_Comp/2_Collision': loss_meta_coll,
            'Meta_Comp/2_1_Collision_Depth': collision_depth.mean(),
            'Meta_Comp/3_Control': loss_meta_ctrl,
            'Meta_Comp/4_HeightBounds': loss_meta_height_bounds,
            'Meta_Comp/6_Stuck': loss_meta_stuck,
            'Meta_Comp/8_Smooth_Jerk': loss_meta_jerk,
            'Meta_Comp/9_Smooth_Snap': loss_meta_snap,
            'Meta_Comp/10_Progress': loss_meta_progress,
            'Meta_Comp/10_1_Progress_Weighted': args.meta_progress_weight * loss_meta_progress,
            'Meta_Comp/11_Energy': loss_meta_energy,
            'Meta_Comp/11_1_Energy_Weighted': args.meta_energy_weight * loss_meta_energy,
            'Meta_Comp/12_Yaw_Smooth': loss_yaw_smooth,
            'Meta_Comp/13_Velocity_Track': meta_loss_velocity_track,
            'Meta_Comp/13_1_Velocity_Track_Weighted': args.meta_velocity_track_weight * meta_loss_velocity_track,
            'Guidance/Applied_In_Current_Phase': 1.0 if guidance_active_this_phase else 0.0,
            'Guidance/Backprop_In_Current_Phase': guidance_backprop_in_phase,

            # === 性能指标 ===
            'Metrics/Success_Rate': success.float().mean(),
            'Metrics/No_Collision_Rate': collision_free.float().mean(),
            'Metrics/Actual_Position_Min_Clearance': actual_dist_obj.min(),
            'Metrics/Lookahead_Min_Clearance': dist_obj.min(),
            'Metrics/Reach_Goal_Rate': reached_goal.float().mean(),
            'Metrics/Arrival_Reward_Hit_Rate': meta_arrival_hit_rate,
            'Metrics/Arrival_Reward_Best_Dist': meta_arrival_best_dist,
            'Metrics/Final_Dist_To_Goal': final_dist_to_goal.mean(),
            'Metrics/Final_Dist_To_Goal_Min': final_dist_to_goal.min(),
            'Metrics/Final_Dist_To_Goal_Max': final_dist_to_goal.max(),
            'Metrics/Avg_Speed': avg_speed,
            'Speed_Command/Mean': command_speed.mean(),
            'Speed_Command/Min': command_speed.min(),
            'Speed_Command/Max': command_speed.max(),
            'Speed_Command/Actual_Mean': v_norm.mean(),
            'Speed_Command/Tracking_Loss': loss_velocity_track,
            'Speed_Command/Prediction_Loss': loss_velocity_predict,
            'Metrics/Speed_Below_Threshold': (avg_speed < min_speed_threshold).float(),
            'Metrics/Min_Speed': v_norm.min(),
            'Metrics/Max_Speed': v_norm.max(),
            'Metrics/Episode_Length': actual_T,
            'Control/Accel_Cmd_Norm_Mean': act_cmd_norm_mean,
            'Control/Accel_Cmd_X_Mean': act_cmd_mean[0],
            'Control/Accel_Cmd_Y_Mean': act_cmd_mean[1],
            'Control/Accel_Cmd_Z_Mean': act_cmd_mean[2],
            'Control/Accel_Cmd_X_AbsMean': act_cmd_abs_mean[0],
            'Control/Accel_Cmd_Y_AbsMean': act_cmd_abs_mean[1],
            'Control/Accel_Cmd_Z_AbsMean': act_cmd_abs_mean[2],

            'Status/Exploration_Window_Effective': exploration_window_effective,
            'Status/Guidance_All_Phases': 1.0 if args.guidance_all_phases else 0.0,

            # === Training map state ===
            'Map/Precomputed_Enabled': 1.0 if args.use_precomputed_geometry_maps else 0.0,
            'Map/Online_Random': 0.0 if args.use_precomputed_geometry_maps else 1.0,
            'Map/Regenerated_This_Iter': float(regenerate_map_now),
            'Map/Current_Index': current_precomputed_map_idx,
            'Map/Type_Code': map_type_code,
            'Map/Curriculum_Stage_Code': map_stage_code,
            'Map/Is_Easy': 1.0 if current_precomputed_map_type == "easy" else 0.0,
            'Map/Is_Hairpin': 1.0 if current_precomputed_map_type == "hairpin" else 0.0,
            'Map/Is_U_Min': 1.0 if current_precomputed_map_type == "u_min" else 0.0,

            # === LGN 六个动态损失权重 ===
            'LGN_Weight/0_Avoidance': weights_eff_mean_tb[0],
            'LGN_Weight/1_Exploration': weights_eff_mean_tb[1],
            'LGN_Weight/2_Turn': weights_eff_mean_tb[2],
            'LGN_Weight/3_Progress': weights_eff_mean_tb[3],
            'LGN_Weight/4_Smoothness': weights_eff_mean_tb[4],
            'LGN_Weight/5_Energy': weights_eff_mean_tb[5],
            'LGN_Output_Stats/Min': weights_raw_tb.min(),
            'LGN_Output_Stats/Max': weights_raw_tb.max(),
            'LGN_Output_Stats/Mean': weights_raw_tb.mean(),
            'LGN_Output_Stats/Std': weights_eff_tb.std(unbiased=False),
            'LGN_Snapshot/0_AvoidanceWeight': weights_snapshot_eff_tb[0],
            'LGN_Snapshot/1_ExplorationWeight': weights_snapshot_eff_tb[1],
            'LGN_Snapshot/2_TurnWeight': weights_snapshot_eff_tb[2],
            'LGN_Snapshot/3_ProgressWeight': weights_snapshot_eff_tb[3],
            'LGN_Snapshot/4_SmoothnessWeight': weights_snapshot_eff_tb[4],
            'LGN_Snapshot/5_EnergyWeight': weights_snapshot_eff_tb[5],
            'LGN_Snapshot/Std': weights_snapshot_eff_tb.std(unbiased=False),
            'LGN_Snapshot/Min': snapshot_raw_flat_tb.min(),
            'LGN_Snapshot/Max': snapshot_raw_flat_tb.max(),
            'LGN_Snapshot/Mean': snapshot_raw_flat_tb.mean(),
        }

        if not args.use_precomputed_geometry_maps:
            log_data.update({
                'Map/Sampled_Easy_Density_Multiplier': env.current_easy_density_multiplier,
                'Map/Sampled_Hard_Density_Multiplier': env.current_hard_density_multiplier,
                'Map/Easy_Density_vs_Old': env._effective_reference_density('easy'),
                'Map/Hard_Density_vs_Old': env._effective_reference_density('hard'),
                'Map/Obstacle_X_Min': env.obstacle_x_min,
                'Map/Obstacle_X_Max': env.obstacle_x_max,
                'Map/Obstacle_Y_Min': env.obstacle_y_min,
                'Map/Obstacle_Y_Max': env.obstacle_y_max,
            })

        if guidance_active_this_phase:
            log_data.update({
                'Meta_Comp/5_Guidance': loss_meta_guidance,
                'Guidance/Dir_Align': guidance_components['dir_align'],
                'Guidance/Overspeed': guidance_components['overspeed'],
                'Guidance/Underspeed': guidance_components.get('underspeed', 0.0),
                'Guidance/Speed_Diff': guidance_components.get('speed_diff', 0.0),
                'Guidance/Escape': guidance_components['escape'],
                'Guidance/Depth': guidance_components['depth'],
                'Guidance/Valid_Ratio': guidance_components['valid_ratio'],
                'Guidance/Collision_Ratio': guidance_components['collision_ratio'],
                'Guidance/Boost': guidance_components.get('guidance_boost', 1.0),
                'Guidance/Sample_Count': guidance_components['sample_count'],
                'Guidance/Avg_Ref_Speed': guidance_components.get('avg_ref_speed', 0.0),
                'Guidance/Avg_Lateral_Error': guidance_components.get('avg_lateral_error', 0.0),
                'Guidance/Max_Lateral_Error': guidance_components.get('max_lateral_error', 0.0),
                'Guidance/Field_Dir_Align': guidance_components.get('field_dir_align', 0.0),
            })

        if train_lgn_phase:
            log_data.update({
                'Grad/LGN_Max_Abs': lgn_grad_max,
                'Grad/LGN_NonFinite_Count': lgn_grad_nonfinite,
                'Grad/LGN_GradElem_Count': lgn_grad_elems,
                'Grad/LGN_Clip_PreNorm': lgn_clip_pre,
                'Grad/LGN_MetaProbe_NonFinite_Count': lgn_meta_probe_nonfinite,
                'Grad/LGN_MetaProbe_GradElem_Count': lgn_meta_probe_elems,
            })
            if gradient_alignment_stats is not None:
                log_data.update({
                    'Diagnostics/Proxy_Meta_Gradient_Alignment_Cosine': gradient_alignment_stats['cosine'],
                    'Diagnostics/Proxy_Meta_Gradient_Dot': gradient_alignment_stats['dot'],
                    'Diagnostics/Proxy_Core_Gradient_Norm': gradient_alignment_stats['left_norm'],
                    'Diagnostics/Meta_Unrolled_Gradient_Norm': gradient_alignment_stats['right_norm'],
                    'Diagnostics/Proxy_Meta_Gradient_Paired_Elements': gradient_alignment_stats['paired_elements'],
                    'Diagnostics/Proxy_Meta_Gradient_Alignment_Usable': gradient_alignment_stats['usable'],
                })
            log_data.update({
                'Loss/4_LGN_Gradient_Alignment': gradient_alignment_loss,
                'LGN/Gradient_Alignment_Cosine': gradient_alignment_cosine,
                'LGN/Gradient_Alignment_Weight': float(args.lgn_grad_alignment_weight),
                'LGN/Gradient_Alignment_Usable': 1.0 if gradient_alignment_usable else 0.0,
            })
            if sparse_temporal_stats is not None:
                log_data.update(sparse_temporal_stats)
        else:
            log_data.update({
                'Grad/Worker_Global_Norm': worker_grad_norm,
                'Grad/Worker_Max_Abs': worker_grad_max,
                'Grad/Worker_NonFinite_Count': worker_grad_nonfinite,
                'Grad/Worker_GradElem_Count': worker_grad_elems,
                'Grad/Worker_Clip_PreNorm': worker_clip_pre,
                'Grad/Worker_Proxy_Norm': worker_proxy_grad_norm,
                'Grad/Worker_Arrival_Norm': worker_arrival_grad_norm,
                'Grad/Worker_Proxy_Scale': worker_proxy_grad_scale,
                'Grad/Worker_ProxyArrival_Cosine': worker_proxy_arrival_grad_cosine,
                'Grad/Worker_ProxyConflict_Projected': worker_proxy_conflict_projected,
            })

        if train_lgn_phase:
            log_data['Loss/3_LGN_Unrolled_Meta'] = lgn_update_loss
            log_data['Meta_Unrolled/1_Position'] = meta_pos_ur
            log_data['Meta_Unrolled/0_Arrival_Reward'] = meta_arrival_reward_ur
            log_data['Meta_Unrolled/0_Arrival_Hit_Rate'] = meta_arrival_hit_rate_ur
            log_data['Meta_Unrolled/0_Arrival_Best_Dist'] = meta_arrival_best_dist_ur
            log_data['Meta_Unrolled/2_Collision'] = meta_coll_ur
            log_data['Meta_Unrolled/3_Control'] = meta_ctrl_ur
            for meta_term_name, meta_term_value in meta_terms_ur.items():
                log_data[f'Meta_Unrolled_Term/{meta_term_name}'] = meta_term_value

        if geom_feat_last is not None and progress_feat_last is not None:
            log_data['LGN_Input/Geom_Mean'] = geom_feat_last.mean()
            log_data['LGN_Input/Geom_Std'] = geom_feat_last.std(unbiased=False)
            log_data['LGN_Input/Geom_Norm'] = geom_feat_last.norm(dim=-1).mean()
            log_data['LGN_Input/Progress_Mean'] = progress_feat_last.mean()
            log_data['LGN_Input/Progress_Std'] = progress_feat_last.std(unbiased=False)
            log_data['LGN_Input/Progress_Norm'] = progress_feat_last.norm(dim=-1).mean()
            for feat_idx in range(min(4, geom_feat_last.shape[-1])):
                log_data[f'LGN_Input/Geom_{feat_idx}'] = geom_feat_last[:, feat_idx].mean()
            for feat_idx in range(min(4, progress_feat_last.shape[-1])):
                log_data[f'LGN_Input/Progress_{feat_idx}'] = progress_feat_last[:, feat_idx].mean()

        active_map_log_key = _resolve_map_log_key(current_precomputed_map_type, map_writers)
        active_tb_writer = _resolve_tb_writer(current_precomputed_map_type, writer, map_writers)
        smooth_dict(log_data, scaler_q_by_map, map_log_key=active_map_log_key)
        if tb_log_now:
            writer_q = scaler_q_by_map[active_map_log_key]
            active_tb_writer.add_scalar("Status_Raw/Train_LGN_Phase", float(train_lgn_phase), i + 1)
            active_tb_writer.add_scalar('Status/Train_Mode', 1.0 if train_lgn_phase else 0.0, i + 1)
            active_tb_writer.add_scalar('Status/Maze_Age', (maze_update_counter - 1) % args.maze_update_interval, i + 1)
            for k, v in writer_q.items():
                active_tb_writer.add_scalar(k, sum(v) / len(v), i + 1)
            writer_q.clear()

        if capture_viz_now:
            idx = 0
            astar_paths_all = guidance_components.get('sampled_astar_paths', [])
            astar_paths_sampled = astar_paths_all[idx] if (isinstance(astar_paths_all, list) and idx < len(astar_paths_all)) else []
            depth_stack = (
                torch.stack(depth_history).detach().to(device='cpu', dtype=torch.float32)
                if len(depth_history) > 0
                else None
            )
            latest_viz_by_map_type[current_precomputed_map_type] = {
                'iter': i + 1,
                'map_type': current_precomputed_map_type,
                'map_idx': int(current_precomputed_map_idx),
                'map_file': str(current_precomputed_map_file),
                'stage': str(current_precomputed_stage),
                'env_snapshot': snapshot_env_for_viz(env, idx=idx),
                'p_cpu': p_history[:, idx].detach().cpu().clone(),
                'v_cpu': v_history[:, idx].detach().cpu().clone(),
                'rpy_cpu': rpy_history[:, idx].detach().cpu().clone(),
                'R_cpu': R_history[:, idx].detach().cpu().clone(),
                'act_cpu': real_act_history[:, idx].detach().cpu().clone(),
                'weights_cpu': effective_weights_seq[:, idx, :].detach().cpu().clone(),
                'depth_stack': depth_stack,
                'astar_paths_sampled': astar_paths_sampled,
            }

        if artifact_save_now:
            save_step = i + 1
            selection_loss = float(meta_loss.detach().item())
            is_new_best = math.isfinite(selection_loss) and selection_loss < best_meta_loss

            torch.save(worknet.state_dict(), os.path.join(save_dir, f'worker_ckpt_{save_step:06d}.pth'))
            torch.save(lgn.state_dict(), os.path.join(save_dir, f'lgn_ckpt_{save_step:06d}.pth'))

            for map_type in VIZ_MAP_TYPES:
                record = latest_viz_by_map_type.get(map_type)
                if record is None:
                    continue
                save_cached_viz_record(
                    record,
                    i + 1,
                    args=args,
                    potential_map_cache=POTENTIAL_MAP_CACHE,
                    video_dir=video_dir,
                    writer=writer,
                )

            if is_new_best:
                best_meta_loss = selection_loss
                best_meta_loss_step = save_step
                best_artifact_label = f'best_step_{save_step:06d}'
                best_worker_file = f'worker_{best_artifact_label}.pth'
                best_lgn_file = f'lgn_{best_artifact_label}.pth'
                torch.save(worknet.state_dict(), os.path.join(save_dir, best_worker_file))
                torch.save(lgn.state_dict(), os.path.join(save_dir, best_lgn_file))

                best_record = latest_viz_by_map_type.get(current_precomputed_map_type)
                if best_record is not None:
                    save_cached_viz_record(
                        best_record,
                        save_step,
                        args=args,
                        potential_map_cache=POTENTIAL_MAP_CACHE,
                        video_dir=video_dir,
                        writer=writer,
                        artifact_label=best_artifact_label,
                    )

                best_metadata = {
                    'step': best_meta_loss_step,
                    'meta_loss': best_meta_loss,
                    'worker_checkpoint': best_worker_file,
                    'lgn_checkpoint': best_lgn_file,
                    'artifact_label': best_artifact_label,
                    'map_type': current_precomputed_map_type,
                    'map_index': int(current_precomputed_map_idx),
                    'map_file': str(current_precomputed_map_file),
                    'map_source': 'precomputed' if args.use_precomputed_geometry_maps else 'online_random',
                }
                with open(os.path.join(save_dir, 'best_checkpoint.json'), 'w') as f:
                    json.dump(best_metadata, f, indent=4)
                writer.add_scalar('Best/Meta_Loss', best_meta_loss, save_step)
                writer.add_scalar('Best/Step', best_meta_loss_step, save_step)
                print(
                    f"[BestCheckpoint] iter={save_step} "
                    f"meta_loss={best_meta_loss:.6f} "
                    f"map={current_precomputed_map_type}:{current_precomputed_map_idx}"
                )

            writer.flush()

print(f"Training Finished. Artifacts in: {save_dir}")
for _map_writer in map_writers.values():
    _map_writer.flush()
    _map_writer.close()
writer.flush()
writer.close()
