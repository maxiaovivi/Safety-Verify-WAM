# Stage 1：AHA-WAM → OVCR-S 动作生成模型

## 任务目标

冻结已训练的 AHA-WAM 教师，把它的最终动作行为蒸馏到 Efficient-WAM-S 尺寸的 OVCR-S 学生。Stage 1 只训练动作生成模型，不训练安全分类头，也不需要风险标签。

学生接收以下张量：

- Efficient-WAM VAE 编码的当前观察 latent tokens；
- 冻结 Efficient 视频骨干真实产生的完整 12 层 K/V，包含当前帧和想象未来；
- Efficient 归一化空间内的当前状态、带噪动作 chunk 和动作时间步。

AHA 只提供最终动作样本。AHA 动作和数据集动作先还原到物理 qpos，再转入 Efficient 的归一化空间。学生输出 16×14 的动作 flow velocity。训练完成后保存的权重只包含 OVCR-S 学生，不包含两个冻结模型。

## 固定网络结构

- 视频 K/V：12 层，2048 维，16 heads × 128；
- 观察查询：32 queries，query dim 512；
- K/V 编辑器：共享低秩编辑器，rank 256，逐层 gate 初始值 -4；
- 动作专家：12 层，hidden dim 768，FFN dim 3072；
- 动作 chunk：16 steps × 14 dims；
- 蒸馏观察层：学生第 `[3, 6, 9, 12]` 层。

参数以 `configs/stage1_ovcr_s.yaml` 为准。服务器实验不得悄悄修改这些尺寸；如因 checkpoint 或数据维度冲突必须修改，需要停止正式训练并报告冲突。

## 代码入口

- 学生网络：`safety_verify_wam.stage1.OVCRSActionGenerator`
- AHA 教师适配：`safety_verify_wam.stage1.AHAOVCRTeacherAdapter`
- AHA Trainer 兼容模型：`safety_verify_wam.stage1.AHAOVCRSStage1Program`
- Efficient 训练输入：`safety_verify_wam.stage1.efficient_training.EfficientStudentTrainingAdapter`
- Hydra 工厂：`safety_verify_wam.stage1.create_aha_ovcr_s_stage1`

`AHAOVCRSStage1Program.training_loss(sample)` 可以直接接入 AHA-WAM 已有的 `Wan22Trainer` 和 RoboTwin 数据加载器。

## 服务器准备

1. 完整克隆本仓库 `main`，记录 commit SHA；不要使用不完整源码副本。
2. 完整克隆官方 AHA-WAM，并记录 commit SHA。所用版本必须包含：
   - `AHAWAMTeacher.rollout_action_latent_states`；
   - `AHAWAM._predict_action_flow_with_video_state`；
   - `LayerwiseChunkKVCacheEditor.build_layer_updated_cache`。
3. 在 AHA 环境中执行 `pip install -e <Safety-Verify-WAM>`。
4. 准备 AHA-WAM checkpoint、完整 Efficient-WAM checkpoint、Wan2.2 权重、RoboTwin 数据集，以及 AHA/Efficient 两套 normalization statistics。
5. 从 AHA 的 `configs/model/ahawam_ode.yaml` 保留完整 teacher 配置，将顶层模型换成 `create_aha_ovcr_s_stage1`；传入 `student_config`、`loss_config` 和 `efficient_conditioning`。后者必须包含 Efficient 部署配置、两套 statistics 路径和 Efficient Python 根目录。
6. Efficient-WAM 动作专家 checkpoint 传给 `efficient_action_checkpoint`。正式训练优先使用该初始化；随机初始化只用于接口检查或对照实验。

AHA 模型配置需要保留 `num_history_frames`，并确保数据集按相同数量提供历史帧。所有路径写进服务器本地配置，checkpoint、数据和输出目录不得加入 Git。

## 训练前检查

先执行一个 batch 的 forward/backward，不保存正式结果。必须记录：

- AHA checkpoint 是否严格加载；
- Efficient 动作专家成功加载、缺失和形状不一致的 tensor 数量；
- observation tokens、完整 12 层 K/V、当前状态和输出动作的实际 shape；
- 教师所有参数 `requires_grad=False`；
- query encoder、K/V editor、动作 block 1/6/12、action decoder 的梯度范数；
- 单卡或每个 rank 的峰值显存、forward 时间和 backward 时间。

预期核心 shape：

| 张量 | Shape |
| --- | --- |
| observation tokens | `[B, S_obs, 48]` |
| compact video K/V | 12 × `[B, S_video, 2048]` |
| student queries | `[B, 1, 32, 512]` |
| updated K/V | 12 × `[B, 1, S_video, 2048]` |
| noisy/predicted action | `[B, 16, 14]` |
| initial state | `[B, 14]` |

出现 shape 错误、NaN/Inf、教师梯度、学生完全没有梯度或 OOM 时，先停止并保存完整错误日志，不要直接降低网络尺寸。

## 训练顺序

### 短训练

- 固定训练和验证样本清单；
- 固定 seed 42；
- 在更新参数前完成一次验证，记为 `step_000000`；
- 训练 100～300 optimizer steps；
- 每 10 steps 记录所有分项损失和梯度范数；
- 每 50 steps 在同一验证子集评估；
- 保存短训练结束 checkpoint。

短训练中总损失、动作 velocity 和 teacher-action 损失应出现下降趋势，preservation 损失不能持续发散。通过后再运行正式训练。

### 正式训练

- 使用互斥的 train/val episode；记录 episode 数和 action chunk 数；
- 从 Efficient 动作专家初始化；
- 优先从 batch size 1 或 2 开始，根据实测显存设置 gradient accumulation；
- 使用 bf16、梯度裁剪 1.0；
- 每次评估使用固定 validation manifest 和固定随机种子；
- 保存 `latest` 和验证集 teacher-action loss 最低的 `best` checkpoint；
- 至少保留 step 0、best、final 三组离线预测结果。

正式训练先跑 seed 42。确认有效后，再用 seed 43、44 复查结果是否稳定。

## 损失含义

默认总损失为：

```text
L = 1.00 L_velocity
  + 1.00 L_teacher_action
  + 0.25 L_ground_truth_action
  + 0.25 L_preservation
```

- `velocity`：以 AHA 最终动作为干净样本，在 Efficient 空间重新加噪后得到的 flow target MSE；
- `teacher_action`：学生一步还原动作和 AHA 最终 rollout 动作的 MSE，二者都在 Efficient 空间；
- `ground_truth_action`：学生一步还原动作和数据集动作的 MSE；
- `preservation`：学生和冻结原 Efficient 动作专家在同一 K/V、状态、噪声和时间步下的 velocity MSE。

AHA query、route、K/V delta 和 response 属于另一套骨干特征空间，不能直接监督 Efficient K/V，因此这些损失在该路线中固定为 0。

## 需要回传的结果

请把下列内容一起返回，不能只发一张 loss 截图：

1. 两个仓库的 commit SHA、当前分支和 `git status`；
2. 完整启动命令和 resolved config；
3. GPU 型号/数量、CUDA、PyTorch、mixed precision、全局 batch size；
4. 数据集名称、train/val episode 数、chunk 数、normalization statistics 文件哈希；
5. `step_000000`、best、final 的全部训练和验证分项指标；
6. 原始逐 step 指标文件，优先 JSONL 或 CSV；
7. query encoder、editor、动作 block 1/6/12、decoder 的梯度范数；
8. 峰值显存、每 optimizer step 时间和样本吞吐量；
9. `best` checkpoint 路径、SHA256 和对应 step；
10. 固定验证样本上的 `predictions.npz`，包含 noisy action、teacher action、ground-truth action、student action、sigma、chunk index 和 sample id；
11. 若已接入多步生成，返回 1/2/4-step 的 student-vs-teacher 和 student-vs-GT action MSE；
12. 失败时返回最早出现异常前后至少 50 行完整日志。

建议同时生成 `summary.json`，至少包含 `initial`、`best`、`final`、`resources`、`data`、`commits` 和 `checkpoint_sha256` 字段。

## 判断训练有效

以下数值是本项目 Stage 1 的建议接受条件，需要在固定验证集上计算：

- 所有损失和梯度保持有限值，没有 NaN/Inf；
- step 10 以后，query encoder、editor、动作早/中/晚层和 decoder 均出现非零梯度；
- best `val_loss_teacher_action` 相比 `step_000000` 至少下降 20%；
- best `val_loss_velocity` 相比 `step_000000` 至少下降 15%；
- `val_loss_preservation` 没有持续发散；
- 固定噪声下的多步 student-vs-teacher action MSE 相比未训练学生至少下降 20%；
- student action 的逐维标准差没有坍缩，建议保持在 teacher 对应标准差的 0.5～1.5 倍；
- 训练集下降但验证集没有改善，不能判定有效。

离线条件通过后，直接使用同一套 Efficient 完整 K/V 部署路径做配对闭环任务测试。Stage 1 的结果只能说明动作行为蒸馏有效，不能说明风险判别已经有效。

## 实验交付

大型 checkpoint、日志、视频和数组保存在 Git 外。请提供 artifact manifest，把每个文件映射到准确的分支、commit、配置哈希、数据 split、seed 和训练 step。实验结束后创建并验证包含全部实验分支的 Git bundle，再把 bundle、manifest 和结果目录一起返回。
