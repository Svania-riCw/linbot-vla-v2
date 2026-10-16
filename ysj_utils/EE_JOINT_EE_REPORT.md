# EE Delta 与 Joint+EE：实现及实验说明

日期：2026-10-16。个人分支：`ysj/action-parameterization`。

已实现数据接入、目标变换、训练损失、推理动作还原和 RoboTwin 控制器分发。本地 CPU 最小测试通过；真实数据、完整模型 GPU 短训和仿真成功率尚未验证，目前没有涨分结论。

## 实现定义

| 实验组 | 训练目标 | 执行接口 | 用途 |
|---|---|---|---|
| B0 | 绝对 Joint + 绝对夹爪 | qpos | 本轮绝对关节对照 |
| JDelta | Joint Delta + 绝对夹爪 | qpos | 本轮关节增量对照 |
| EEDelta | EE Delta + 绝对夹爪 | ee，走 RoboTwin IK/规划 | 验证末端空间控制 |
| JointEE0 | Joint Delta + EE Delta，EE 监督权重0 | qpos | 相同联合目标表示的零权重消融 |
| JointEE | Joint Delta + EE Delta，EE 监督权重默认0.1 | qpos | 验证末端辅助监督 |

EE Delta 的所有未来目标都以**当前观测**为参考，不逐步累加。每只手使用：

```text
delta_position[k] = R(current)^T * (position_target[k] - position_current)
delta_quaternion[k] = inverse(q_current) * q_target[k]
```

训练特征是双臂 `xyz + quaternion_xyzw` 共14维，复用现有 `quaternion_local` 变换。推理先反归一化并还原到世界坐标，再转成 RoboTwin 的 `[xyz, quaternion_wxyz, gripper] × 2` 共16维，显式调用 `take_action(..., action_type='ee')`。夹爪保持绝对目标。

Joint+EE 使用现有55维 flow head 同时预测：12关节 + 14末端 + 2夹爪，共28个有效通道。执行时固定使用还原后的14维 joint/gripper，EE 输出作为辅助预测。没有新增独立 head，也没有可微 FK 或两个输出之间的运动学一致性约束。

```text
L_action = mean_valid(L_joint_and_gripper) + lambda * mean_valid(L_ee)
L_total  = L_action + 原有 depth / future depth / video / MoE 等损失
```

两组分别取均值，避免新增通道后因分母变大而减弱关节监督。`lambda=0.1` 是待验证初值，不是已验证最优值。日志增加 `action/joint_gripper_loss` 和 `action/ee_loss`。

## 要验证什么，以及如何控制变量

- **EEDelta 假设**：末端增量直接表达位置和姿态变化，可能更适合操作任务。与 JDelta 比较时控制器也改变了，成功率差异应解释为“表示 + 控制器”的整体效果；失败时须检查 IK/规划错误和耗时。
- **JointEE 假设**：保留 joint 执行接口，增加末端监督可能有利于共享特征的空间理解。先比较 JointEE 与 JDelta，再比较 JointEE 与 JointEE0，判断 EE 监督是否额外有效。
- **相同输入**：本轮五组都输入当前 joint、夹爪、EE 状态及原图像/语言。EE 状态也进入模型。旧版两组实验没有 EE 状态输入，不能直接拿旧 checkpoint 当本轮因果对照，需要重新训练控制组。
- **相同 padding 处理**：五组都启用 `mask_action_padding=True`，排除补齐时间步。原两组生成方式默认保持关闭，因此也不能混用旧结果。
- **相同统计**：不同动作表示各自计算 norm；JointEE0 和 JointEE 表示完全相同，后者复制前者经过核验的数值统计，分别保存配置契约。

JointEE0 仍保留 EE 目标及其 flow 输入通道，因此与 JDelta 的条件不同。它隔离相同联合表示下的**EE 监督权重**，不能宣称 lambda=0 时整个训练等同于 JDelta。以上均为实验假设，不保证提升。

## 数据接入与时序审计

`ysj_utils/prepare_ee_labels.py` 读取原始 RoboTwin HDF5 和实际 LeRobot Parquet。使用队内 `xsy_utils/d4rt_data/prep_ee.py` 的 world/tool pose 定义，并额外审计动作目标。

每个 episode 必须满足：frame_index 从0连续且不重复；LeRobot有N行、原始数据有N+1帧；`state[t] == raw_joint[t]`；`action[t] == raw_joint[t+1]`；EE和joint夹爪一致；数值有限、四元数单位长度。位置/关节比较使用绝对容差1e-5，四元数长度容差1e-3。

校验失败时该数据集不发布 manifest。若实际 action 是另一种控制目标，不能直接用观测到的下一帧 EE 代替；应使用匹配机器人模型的 FK 重建目标后另行支持。

标签存入独立 episode NPZ，不改写原始数据。训练和统计共享加载器：当前EE取raw[t]；动作块第k项取raw[t+1+k]；episode尾部与LeRobot一样重复最后目标并加padding。每次加载再核验真实state、action chunk和padding，NPZ受SHA256保护，每个worker最多缓存两个episode。

输出目录包含服务器数据集的绝对路径，应在实际训练服务器生成。实验配置绑定标签manifest；推理继续核验训练机器人配置、统计、动作模式、chunk和norm类型。

## 操作顺序

以下命令在团队Linux训练环境执行，所有路径替换为真实路径。默认训练清单是团队50个clean任务；7个任务是评测范围。若改成7任务训练，所有组应使用统一清单。

### 1. 最小测试

在独立CPU测试环境安装测试requirements；训练环境沿用团队依赖。

```bash
python -m pip install -r tests/action_parameterization/requirements.txt
python tests/action_parameterization/run_checks.py --require-torch --report .runtime/ee_checks.json
```

### 2. 生成并审计标签

```bash
python ysj_utils/prepare_ee_labels.py \
  --train-list /path/to/robotwin_clean.txt \
  --raw-root /path/to/RoboTwin/data \
  --raw-setting aloha-agilex_clean_50 \
  --output /path/to/ee_labels
```

原始文件路径是 `<raw-root>/<task>/<raw-setting>/data/episode<id>.hdf5`。输出中每个数据集子目录必须不存在；中断或失败后请选择新输出目录重新生成。

### 3. 生成五组配置并计算统计

```bash
python ysj_utils/action_experiment.py prepare \
  --base-config /path/to/server_baseline.yaml \
  --robot-config /path/to/robot_config_clean/robotwin.yaml \
  --train-list /path/to/robotwin_clean.txt \
  --ee-sidecar-root /path/to/ee_labels \
  --ee-weight 0.1 \
  --output /path/to/ee_experiments

for group in B0 JDelta EEDelta JointEE0 JointEE; do
  python ysj_utils/action_experiment.py norm --bundle /path/to/ee_experiments --group "$group" || break
done
```

正式配置：全局BS128、10000 optimizer steps、bf16、chunk50、seed42、RTC关闭；默认8卡 × micro batch16 × 梯度累积1，保留原有其他模型设置。JointEE0统计必须先完成，JointEE复用其数值统计并写自己的契约。生成目录中有每组 `robot_configs/robotwin.yaml`、`train.yaml`、`norm.yaml` 及整体 `pair.json`。

### 4. 每组先做5步短训和1条短评测

先完成JDelta控制组，再分别完成EEDelta、JointEE0、JointEE。以下以EEDelta为例：

```bash
CUDA_VISIBLE_DEVICES=0 python ysj_utils/action_experiment.py train \
  --bundle /path/to/ee_experiments --group EEDelta --stage smoke

python ysj_utils/evaluate.py --mode smoke \
  --checkpoint /path/to/ee_experiments/EEDelta/train-smoke/checkpoints/global_step_5/hf_ckpt \
  --gpus 0
```

人工确认loss/梯度有限、step5保存且能重载、末端数值/四元数/双臂夹爪顺序正确、没有持续IK或规划失败、结果文件和视频完整。5步BS1仅排错，不用于效果比较。

### 5. 正式训练和350条评测

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ysj_utils/action_experiment.py train \
  --bundle /path/to/ee_experiments --group EEDelta --stage full

python ysj_utils/evaluate.py --mode full \
  --checkpoint /path/to/ee_experiments/EEDelta/train/checkpoints/global_step_10000/hf_ckpt \
  --gpus 0,1,2,3,4,5,6,7 --seed 0
```

每组同预算。评测复用 `xsy_utils/eval_baseline.sh`：7任务 × clean/random × 25 episodes = 350条/checkpoint，FP32推理，执行chunk长度50。五组全部完成共1750条。代码不会自动启动以上训练。时间紧时先验证JDelta/EEDelta，再做联合目标及零权重消融。

```bash
python ysj_utils/compare_results.py \
  --baseline-run /path/to/JDelta/eval/run_xxx \
  --candidate-run /path/to/EEDelta/eval/run_xxx \
  --output ee_vs_joint.json

python ysj_utils/compare_results.py \
  --baseline-run /path/to/JointEE0/eval/run_xxx \
  --candidate-run /path/to/JointEE/eval/run_xxx \
  --output ee_supervision_ablation.json
```

实际评测目录见命令输出和 `external/RoboTwin/eval_result/`。比较器检查共同输入、标签来源、训练预算、初始模型、评测seed及代码commit；JointEE0/JointEE额外要求相同数值统计。结果为描述性差异，仍须核对场景一致性、失败视频并补seed复核。

## 测试证据与边界

本轮实际结果：**57项通过、0失败、0跳过**，使用 `--require-torch`。环境为Windows、Python3.13、PyTorch2.8.0 CPU，新增pyarrow26.0.0和h5py3.16.0用于真实文件格式测试。20个变更Python文件语法检查及Git空白检查通过。报告的head是测试前基础提交，source_sha256对应实际测试的工作区文件。

本轮记录见 `reports/ee_joint_ee_minimal_2026-10-16.json`，涵盖合成Parquet/HDF5真实文件读写、错位标签拒绝、NPZ校验/worker序列化、实际loader接入、真实动作变换/归一化/padding/服务端逆变换、SciPy独立旋转核对、控制器分发、真实policy forward损失归约、EE梯度、lambda=0、padding和一次优化器更新。

测试隔离图像/语言处理、LeRobot依赖构造和完整VLM。前向归约使用可控预测张量替代重模型内部计算；合成数据用于逻辑检查，不能证明真实FK关系。尚未运行真实标签审计、完整模型bf16前向反向、torch.compile、多GPU、checkpoint重载或RoboTwin IK与350条评测。

本次没有加入通用Weighted Flow-matching或Spline Action。JointEE的lambda是该路径的组间监督权重；后续可独立设计通用加权，避免多项改动混成一组结果。

## 可直接汇报

> EE Delta和Joint+EE两条路径已经接入训练和评测。EE Delta以当前时刻为参考预测末端局部增量，还原后走RoboTwin EE控制器；Joint+EE同时预测关节与末端，保持joint执行，增加独立归约的EE损失，并准备零权重消融。五组有相同状态输入，正式预算是BS128、10000步、bf16。代码包括标签时序审计、统计绑定和训练推理一致性检查，本地最小测试通过；还没有真实GPU或仿真结果，暂时没有涨分结论。下一步先审计服务器数据，完成5步短训和1条仿真，再启动正式对照。
