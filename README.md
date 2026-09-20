# RL Drone SRU

这是一个基于可微物理的视觉无人机飞行训练项目。项目在原有 Worker/LGN 双层优化框架中加入了空间增强循环单元（Spatially Enhanced Recurrent Unit，SRU），用于从深度图和飞行状态中学习带时序记忆的控制策略。

SRU Worker 在整个时序过程中保留固定大小的二维空间记忆，并使用局部卷积、空洞卷积和门控更新融合当前观测与历史信息。训练入口同时保留了因果 Transformer Worker，便于进行消融实验和性能对比。

## 主要功能

- 基于深度图的端到端无人机控制
- 固定大小二维空间循环记忆
- SRU 与因果 Transformer 两种 Worker 骨干网络
- Worker 与损失生成网络（LGN）交替训练
- 可微内循环和 meta rollout
- 在线随机地图与预生成地图训练
- 可选 A* 或 Dijkstra 势场引导
- 训练指标、轨迹和可视化结果记录
- 模型、损失、梯度和 checkpoint 兼容性测试

## 环境要求

CUDA 扩展已在以下环境中测试：

- Python 3.11
- PyTorch 2.2.2
- CUDA 11.8
- Linux

建议使用独立的 Conda 或 Python 虚拟环境。除 PyTorch 外，训练脚本还会使用 NumPy、Matplotlib、TensorBoard、tqdm、Plotly 和 imageio 等常用依赖。

## 编译 CUDA 仿真扩展

项目训练依赖 `quadsim_cuda`。在仓库根目录执行：

```bash
export CUDA_HOME=/usr/local/cuda
PIP_NO_BUILD_ISOLATION=1 pip install ./src --no-build-isolation --no-use-pep517
python -c "import torch; import quadsim_cuda; print(torch.__version__)"
```

如果遇到 `libc10.so` 无法加载，可临时补充 PyTorch 动态库路径：

```bash
export TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
python -c "import torch; import quadsim_cuda"
```

更多编译说明见 [`src/README.md`](src/README.md)。

## 开始训练

SRU 是当前默认 Worker 骨干网络：

```bash
python mmgj_transformer.py \
  --worker_backbone sru \
  --sru_hidden_channels 96 \
  --exp_name sru_run
```

使用 Transformer Worker 作为对照：

```bash
python mmgj_transformer.py \
  --worker_backbone transformer \
  --worker_max_seq_len 32 \
  --exp_name transformer_run
```

一个适合检查训练链路的短程运行示例：

```bash
python mmgj_transformer.py \
  --worker_backbone sru \
  --num_iters 12 \
  --batch_size 2 \
  --timesteps 4 \
  --lgn_timesteps 4 \
  --lgn_steps 1 \
  --worker_steps 1 \
  --artifact_save_interval 0 \
  --guidance_backend none
```

完整的训练参数、地图设置和势场引导方法见 [`使用说明_训练与势场引导.md`](使用说明_训练与势场引导.md)。

## 运行测试

```bash
python -m unittest discover -s tests -v
```

当前测试覆盖：

- SRU 空间记忆形状与梯度传播
- Transformer 时序记忆
- LGN 输出和动态损失权重
- 一阶、二阶及 meta 梯度
- 碰撞、到达、平滑、能耗和偏好损失
- Worker 梯度合并
- checkpoint 输出维度兼容映射

## 目录结构

```text
.
|-- WorkNet_sru.py                 # 空间增强循环 Worker
|-- WorkNet_transformer.py         # 因果 Transformer Worker
|-- LossGenNet_transformer.py      # 动态损失生成网络
|-- mmgj_transformer.py            # Worker/LGN 交替训练入口
|-- env.py                         # 主要可微仿真环境
|-- env_multi.py                   # 多无人机和地图环境
|-- potential_map_utils.py         # 势场查询工具
|-- precompute_potential_maps.py   # 预生成地图和势场
|-- worker_context_features.py     # Worker 上下文特征
|-- configs/                       # 示例参数配置
|-- src/                           # CUDA/C++ 可微仿真扩展
|-- tests/                         # 单元测试
`-- utils/                         # rollout、规划、日志和张量工具
```

## 模型输出

训练过程可保存以下内容：

- Worker 和 LGN 权重
- TensorBoard 日志
- 最优 checkpoint 元数据
- 深度视频和三维轨迹可视化
- 用于复现实验的源码快照

这些运行产物可能较大，默认不会提交到 Git。仓库同样不包含预训练权重、虚拟环境或预编译 CUDA 二进制文件。

## 注意事项

- 正式训练前应先完成 CUDA 扩展编译并运行全部测试。
- `--resume_worker` 对应的 checkpoint 必须与所选 Worker 骨干结构匹配。
- 短程 smoke test 只能验证训练链路，不能代表策略已经收敛。
- 比较 SRU 和 Transformer 时，应固定随机种子、地图集合和训练预算。

## 许可证

本项目使用仓库中 [`LICENSE`](LICENSE) 文件所述许可证。
