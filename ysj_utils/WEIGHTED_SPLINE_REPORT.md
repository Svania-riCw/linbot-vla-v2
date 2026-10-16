# Weighted Flow-matching 与 Spline Action：实现与实验入口

日期：2026-10-16；个人分支 `ysj/action-parameterization`。

两项已经接入训练代码，并准备独立实验配置。Spline 同时接入了实际 flow 采样函数，输出还原成50步动作后复用队内评测。当前完成的是代码与CPU验证，没有GPU训练或机器人任务成功率结果。

## 1. Weighted Flow-matching

在原有逐元素 flow 误差上增加动作维度权重和未来时间步权重：

```text
w[b,k,d] = valid_mask[b,k,d] * dimension_weight[d] * horizon_weight[k]
L_action = sum(w * flow_error) / sum(w)
```

保留原 `L1_fm` 或 `fm` 的误差定义；默认团队配置为L1。这里的时间步k是未来动作位置，不是flow噪声时间s；不改变噪声时间采样分布。补齐时间步、无效动作维度和权重为0的项不参与归约。分母使用有效权重总和，整体缩放权重不会改变损失尺度；全1权重应复现普通masked loss。

配置入口：`train.flow_dimension_weights` 长度55，`train.flow_horizon_weights` 长度50，可分别设置；不设置时保持普通损失。负数、非有限值、长度错误、没有有效监督会报错。

首版提供两个启发式候选，不宣称已验证最优：

- **WGrip**：夹爪通道28、29权重2，其他有效通道权重1。检验更重视夹爪监督能否减少抓取/释放错误。权重基于规范化坐标，不能解释成物理量单位之间的直接比较。
- **WTime**：未来第k步权重从1线性降至0.5。检验强调近期动作能否改善执行；当前仍执行50步，后半段精度可能受损，需要观察失败视频。

另有 **WUniform**：所有权重为1，用作实现等价性检查。CPU已验证普通损失与全1权重一致，也覆盖梯度、padding和整体缩放不变性。一般无需为它额外投入一次正式10000步训练。

实现位于 `lingbotvla/action_spaces/loss.py`，已经从实际 `LingbotVlaV2Policy.forward` 调用；不再只存在于NumPy参考测试中。原有depth/video/MoE损失保持原权重。

## 2. Spline Action

首版使用固定节点、三次clamped B-spline，默认以16个控制点表示原50步动作。动作维度保持55，保留现有输入/输出投影层，因此没有为控制点新增需要从头训练的投影头。

```text
训练：绝对joint/gripper标签
  → joint减去当前状态，夹爪保持绝对
  → 使用原动作统计归一化，形成 [B,50,55]
  → 拟合样条控制点 [B,16,55]
  → 在控制点空间构造flow噪声、目标速度和loss

推理：在 [B,16,55] 控制点空间执行flow采样
  → B-spline解码为 [B,50,55]
  → 反归一化、joint加回当前状态
  → 14维joint/gripper交给RoboTwin qpos控制器
```

因此，这条路径是学习控制点表示，不只是对普通策略输出做平滑。数据chunk和执行长度仍为50，模型动作token数为16；压缩的是动作token，视觉/语言token数量不变。不能据此直接声称整体推理快50/16倍。

完整chunk使用最小二乘拟合。episode尾部只使用有效标签，不使用重复padding；短前缀拟合加入1e-3 ridge，避免病态矩阵放大控制点数值。没有被有效帧支持的控制点不计入loss。矩阵以FP32计算，并在实际设备上缓存，避开模块bf16和meta初始化对拟合矩阵的影响。

配置：`train.spline_control_points=16`、`train.spline_degree=3`；0个控制点表示关闭。节点为固定均匀clamped节点。首版仅支持Joint Delta + 绝对夹爪；不直接插值四元数，不叠加RTC、JointEE或加权loss。

**夹爪也被样条表示**，快速开合可能被平滑；这是必须检查的限制。训练日志增加：

- `spline/target_rmse`：规范化目标投影后的RMSE；
- `spline/target_max_error`：规范化目标最大误差；
- `spline/gripper_target_mae`：规范化夹爪目标平均误差。

这些是表示压缩误差，不是模型预测误差或任务成功率。实现不保证跨chunk连续，不包含可微FK、速度/加速度控制或关节限位约束。

思想参考 [ABPolicy](https://arxiv.org/abs/2602.23901) 的B-spline控制点flow表示，以及 [Spline Policy](https://arxiv.org/abs/2606.07386) 的样条参数化动作；当前为简化实现，不是完整论文复现，也没有实现其异步、双向预测或纠正控制机制。数值基函数与 [SciPy BSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html) 独立对照测试。

## 3. 对照方案

使用 `prepare --family supervision` 生成以下五个目录；不会自动开始训练：

| 组别 | 相对本轮JDelta的改动 | 建议用途 |
|---|---|---|
| JDelta | 无 | 正式控制组 |
| WUniform | 全1维度/时间权重 | 先做短训等价性排错 |
| WGrip | 夹爪权重2 | 首选加权候选 |
| WTime | 近期1、远期0.5 | 时间加权候选，可后续追加 |
| Spline | 16个三次样条控制点 | 动作表示候选 |

时间紧时先正式训练 **JDelta、WGrip、Spline三组**，每组350条评测，共1050条；WTime后续按资源追加。各候选分别与JDelta比较，不把weighted+spline同时打开。

所有组：全局BS128、10000 optimizer steps、bf16、相同初始模型与数据、seed42、50步物理动作、相同joint状态和图像语言输入、padding loss mask开启、RTC关闭。其他depth/video配置沿用baseline。

该family暂时统一关闭torch.compile，先确认eager训练；所有组均关闭以控制变量。它不加入EE状态，因此不能直接混用此前含EE输入的JDelta控制组；早期不屏蔽padding的JDelta也不能直接复用。应在这个bundle内重新建立匹配对照。

五组使用同一份**数值相同**的归一化统计：先计算JDelta，再复制给其他组，分别写配置凭据。Spline统计定义在拟合前的动作空间，系数不再额外归一化。推理会核对保存的权重设置和样条参数，避免把16点模型当50点模型解码。

## 4. H20上的执行顺序

在团队配置好的Linux训练环境中执行，替换实际路径。生成目录必须不存在；不要直接使用Windows预览中的绝对路径。

```bash
python ysj_utils/action_experiment.py prepare \
  --family supervision \
  --base-config /path/to/server_baseline.yaml \
  --robot-config /path/to/robot_config_clean/robotwin.yaml \
  --train-list /path/to/robotwin_clean.txt \
  --spline-points 16 \
  --output /path/to/supervision_experiments

for group in JDelta WUniform WGrip WTime Spline; do
  python ysj_utils/action_experiment.py norm \
    --bundle /path/to/supervision_experiments --group "$group" || break
done
```

本轮不需要EE HDF5标签。新增真实Parquet诊断在CPU环境即可运行，需要numpy、torch、pyarrow、PyYAML；不加载VLM、不读视频、不需要GPU：

```bash
python ysj_utils/check_spline_fit.py \
  --bundle /path/to/supervision_experiments \
  --samples 128 --output /path/to/spline_fit_128.json
```

脚本按固定seed跨训练数据集抽取chunk，保持episode边界，报告规范化误差、关节弧度误差、夹爪误差和快速开合次数。物理量诊断目前要求团队默认的 `bounds_99_woclip`。没有通用通过阈值；需要重点看接触和夹爪切换是否丢失。若K16误差不可接受，应尝试较多控制点并生成新的bundle，不在现有bundle中手改配置。

先对控制与候选做5步短训，以Spline为例：

```bash
CUDA_VISIBLE_DEVICES=0 python ysj_utils/action_experiment.py train \
  --bundle /path/to/supervision_experiments --group Spline --stage smoke

python ysj_utils/evaluate.py --mode smoke \
  --checkpoint /path/to/supervision_experiments/Spline/train-smoke/checkpoints/global_step_5/hf_ckpt \
  --gpus 0
```

检查loss/梯度有限、表示误差、checkpoint保存重载，以及推理实际输出50步动作。小评测正常后再投入正式预算：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ysj_utils/action_experiment.py train \
  --bundle /path/to/supervision_experiments --group Spline --stage full

python ysj_utils/evaluate.py --mode full \
  --checkpoint /path/to/supervision_experiments/Spline/train/checkpoints/global_step_10000/hf_ckpt \
  --gpus 0,1,2,3,4,5,6,7 --seed 0
```

对JDelta和WGrip执行相同步骤，替换组名和checkpoint目录。评测仍复用队内 `xsy_utils/eval_baseline.sh`，7任务×clean/random×25条=350条，FP32推理、执行长度50。

```bash
python ysj_utils/compare_results.py \
  --baseline-run /实际JDelta评测目录/run_xxx \
  --candidate-run /实际Spline评测目录/run_xxx \
  --output spline_vs_joint.json
```

比较器核对共享统计、状态输入、监督配置、训练预算和评测协议。损失数值的降低不能代替成功率比较；Spline还需记录真实推理耗时及轨迹/夹爪误差。当前没有改善成功率或推理速度的实测结论。

## 5. 本地最小验证

最终结果：**72项全部通过，0失败、0跳过**（原57项，加本轮15项）。环境为Windows/Python3.13/PyTorch2.8.0 CPU；14个变更Python文件的语法检查通过。报告中的head是测试前基础提交，source_sha256记录实际测试工作区源码。

```bash
python tests/action_parameterization/run_checks.py \
  --require-torch --report .runtime/supervision_checks.json
```

测试覆盖：权重归约与独立NumPy参考、全1等价性、梯度、无效权重拒绝；基函数与SciPy对照、控制点往返、尾部padding隔离、bf16数值路径；实际flow训练forward/速度预测/Euler采样与50步解码，再经过真实Normalizer与Joint Delta逆变换进入qpos控制器。还覆盖Parquet抽样、夹爪跳变误差、配置/统计契约和独立评测比较。

完整视觉语言模型、真实LeRobot依赖构造、CUDA/FSDP、checkpoint重载与仿真仍未执行。测试中的视觉语言嵌入和attention主干使用替身，不可把CPU测试写成“全模型训练已跑通”。原始记录见 `reports/weighted_spline_minimal_2026-10-16.json`。

## 可直接汇报

> 我补充实现了通用Weighted Flow-matching和Spline Action。前者可分别调整动作维度与未来时间权重，并有全1等价性检查；后者用16个三次B样条控制点学习50步Joint Delta轨迹，训练和推理解码都已接入。准备了JDelta对照、夹爪加权、时间加权和Spline候选，还增加真实Parquet压缩误差诊断。先建议JDelta/WGrip/Spline三组同预算对照。当前完成CPU最小验证，真实H20训练和350条仿真尚未完成，没有涨分或加速结论。
