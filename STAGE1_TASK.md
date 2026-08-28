# Stage 1：Efficient 尺寸的小 AHA 动作生成器

## 当前目标

Stage 1 先验证一个结构对齐的、小尺寸 AHA ActionDiT 能否读取 AHA 当前观察 K/V、任务文本和当前机器人状态，并在完整 RoboTwin 数据上学习 16×14 动作 flow。安全分类、风险标签和小视频骨干不属于这一阶段。

旧运行 `aha-current-gt-action-full80k-b16-seed42` 在 step 36000 停止。它保留为 Efficient 风格联合注意力的比较组，不再续训，因为其动作层缺少 AHA 的独立上下文注意力。新结构从 step 0 计数；旧权重只用于初始化兼容部分。

## 固定尺寸

- 12 个动作层；action hidden 768；FFN 3072；
- 16 heads × 128 head dim，联合注意力空间为 2048；
- 动作 chunk 为 16×14；
- 32 个 observation queries，query dim 512；
- 共享 rank-256 K/V editor；
- 任务文本与 proprio context 的输入维度为 4096。

尺寸以 `configs/stage1_ovcr_s.yaml` 为准。正式运行不得静默修改。

## AHA 对齐的数据流

每个动作层执行：

```text
noisy action --Linear--> action tokens --1D RoPE--+
                                                     | mixed attention
AHA current-frame K/V --OVCR-S editor---------------+
                                                     |
task T5 tokens + current proprio token               | residual
                 --text embedding--independent cross-attention
                                                     |
                                                   FFN
```

必须满足：

- action encoder 是一层 `Linear(14, 768)`；
- 没有 state token、register token或固定位置 embedding；
- mixed attention 使用独立且带 bias 的 `q/k/v/o`；
- action query/key 使用与 AHA 相同的 1D RoPE；
- 每层包含 `norm3 -> cross_attn -> residual`；
- task T5 context 与当前 14 维 proprio 经 4096 维 context 路径进入 cross-attention；
- action head 是 `Linear(768, 14)`，不附加 Efficient decoder 的 AdaLN；
- 单个 16-step chunk 内为双向动作注意力。

`ObservationQueryEncoder` 与共享 rank-256 editor 是本项目的小型化设计，不声称与完整 AHA editor 参数结构相同。

## 初始化

正式候选使用两种明确来源：

1. 旧 step 36000 只加载 query encoder 与 K/V editor；
2. 完整 AHA action expert 做确定性结构切片：30→12 层、24→16 heads、1024→768 hidden、4096→3072 FFN；
3. shape 完全相同的 AHA proprio encoder 直接复制；
4. 每个 checkpoint 保存初始化来源、层映射、选择规则和旧 checkpoint 路径。

旧动作权重迁移路线只作为短测对照。它需要丢弃旧 state/register 数据流，并改变 action encoder 与 head，因此不能默认视为连续恢复。

## 训练输入与损失

- AHA 只执行当前观察视频 prefill，提供原始 48 维观察 tokens、映射后的 12 层当前帧 K/V、任务 T5 context；
- 当前 proprio 从所选动作 chunk 起点读取，和 AHA 使用同一份归一化数据；
- 不运行无用的 16-step 教师动作去噪；
- 完整 RoboTwin 数据，1000 档 uniform shifted noise，shift 5；
- 当前损失为 `1.0 × GT flow velocity`；
- action expert、proprio encoder、query encoder 和 K/V editor 一起训练；
- AHA 教师参数始终冻结。

## 开跑前检查

必须全部通过：

- 单元测试确认动作 token 数恰好为 16，cross-attention、RoPE、状态和文本路径存在；
- AHA structured slice 初始化覆盖 action expert 的每个 tensor；
- 旧 checkpoint 的 conditioning-only 加载没有形状冲突；
- 一个真实 batch 前向、反向和 optimizer step 都为有限值；
- step 2 后 proprio、text embedding、cross-attention、早中晚 action block、query encoder、editor 均有非零梯度；
- 训练和 RoboTwin policy 都传入相同的 task context、context mask 与当前 proprio；
- 记录参数量、峰值显存、data/fwd/bwd/optimizer 时间。

## 短测与继续条件

- step 0：固定验证面板与动作分布；
- step 50/200：检查学习方向、梯度与数值；
- step 2000：固定面板、多步动作去噪和小规模 RoboTwin 任务测试；
- 只有固定验证集 GT action MSE 改善、动作方差未坍缩，并且至少一个任务/seed 出现相对改善，才继续更长训练；
- 结构切片初始化与旧动作迁移使用相同数据、seed、batch 和验证面板做短测，不混记结果。

正式训练从 step 0 开始，checkpoint 间隔不小于 2000 steps，避免磁盘被重复权重占满。大型权重、日志、数组和视频放在 Git 外，artifact manifest 必须记录 branch、commit、配置哈希、task、setting 和 seed。

## 当前尚未包含

Stage 1 的任务测试仍使用完整 AHA 视频 prefill。最终可独立部署的小 AHA 还需要：

- 12 层、2048 hidden 的小视频骨干及其训练；
- 需要时加入跨 chunk action history；
- 对小视频骨干与动作专家做联合训练和完整任务评估。

因此 Stage 1 通过只说明“小 AHA 动作路径有效”，不能说明整个小 AHA 已完成。
