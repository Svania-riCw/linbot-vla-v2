# 姚苏珈：动作表示实验

最新更新：通用Weighted Flow-matching与Spline Action已接入，见 [加权与样条实验说明](WEIGHTED_SPLINE_REPORT.md)。包含独立配置、真实Parquet压缩误差检查和现有evaluation接入。

2026-10-16更新：EE Delta、Joint+EE实现、五组匹配对照与服务器命令见 [EE / Joint+EE说明](EE_JOINT_EE_REPORT.md)。以下保留第一阶段Joint Delta说明；两阶段状态输入和padding设置不同，不能混用checkpoint做因果对照。

状态：**开发和本地检查版本，尚未通过 GPU 训练及 RoboTwin 实测。**

2026-10-16 最小测试结果：**44项全部通过，0失败、0跳过**。环境为Windows/Python 3.13/PyTorch 2.8.0 CPU；包括此前未执行的张量动作路径、EE梯度和裁剪归一化测试。报告见 [第一阶段汇报](REPORT_2026-10-16.md) 和 [测试原始记录](reports/action_minimal_2026-10-16.json)。

基于团队 main `56952d10a8ce9c1224e7c639c17fae92dfe78a94` 开发，个人分支 `ysj/action-parameterization`。此版本先交付可审查的实现和实验入口，没有成功率提升结论。

## 本次完成

1. 迁移旧动作表示测试，并核对新仓库的动作转换、归一化和机器人配置。迁移前四个相关文件与旧上游一致；当前分支随后修正了下面的默认类型问题。
2. 修复 joint delta 配置问题：原默认 `quaternion_local` 会让 joint 的 `subtract_state=True` 误入 EE 四元数分支。现在仅 `end.position` 默认使用该类型，joint 使用普通差值。新增配置解析回归测试。
3. 修复评测绑定问题：不仅加载训练用 norm_stats，还把训练的 robot_config_root 传给推理动作变换，确保 joint delta 在推理时加回当前状态。保留绝对动作默认行为。
4. 增加 B0/JDelta 成对配置生成、各自重新计算 norm_stats、归一化来源记录、5步短训与正式训练入口。
5. 增加小规模/正式评测入口、结果完整性检查和两组结果比较。底层复用队内的 xsy_utils/eval_baseline.sh。

## 固定实验标准

| 项目 | B0与JDelta共同设置 |
|---|---|
| 训练 | 全局 BS128、10000 optimizer steps、bf16 |
| 默认硬件映射 | 8 GPU × micro batch16 × 梯度累积1；可调整但必须保持全局128 |
| 动作 | chunk50，夹爪始终绝对；JDelta所有未来关节目标减去同一个当前观测状态 |
| 其他改动 | RTC关闭，不叠加EE辅助loss、weighted loss、spline或VGGT |
| 正式评测 | 7任务 × clean/random × 每格25条 = 350条/checkpoint |
| 任务ID | blocks_ranking_rgb、hanging_mug、open_laptop、pick_diverse_bottles、rotate_qrcode、stack_blocks_three、place_shoe |
| 评测精度 | 两组沿用队内默认FP32推理；训练仍是bf16。不得只给一组改评测精度 |
| 训练数据 | 暂沿用新仓库50个clean任务清单；需队内确认，7个评测任务不等于只训练7个任务 |

已有5000步FP32 baseline只能用来检查评测流程，不能直接当本轮10000步bf16的匹配对照。

## 本地测试

```bash
python tests/action_parameterization/run_checks.py --report .runtime/action_checks.json
```

依赖 NumPy、SciPy、PyYAML。报告不会覆盖已有文件，请换新文件名。在团队已有PyTorch环境执行：

```bash
python tests/action_parameterization/run_checks.py --require-torch --report .runtime/action_checks_torch.json
```

`--require-torch` 有任何跳过项即失败。测试分为独立数值参考、执行实际源函数、实际张量字段映射/padding；图像/语言处理与模型前向不在本地测试覆盖范围。

真实数据检查脚本在 `tests/action_parameterization/check_dataset.py`。输入NPZ字段为 `observed_state[N,14]`、`actions[N,H,14]`、`action_is_pad[N,H]`、`episode_id[N]`、`action_episode_id[N,H]`。关节为rad，布局为左6关节、左夹爪、右6关节、右夹爪。需从实际转换器导出，不能用归一化后的batch冒充原始数据。示例/合成数据通过不能当作真实数据已验证。

## 服务器执行顺序

需完整源码检出、队内已配置的 lingbotvla/robotwin-eval 环境、模型权重和仿真资源。先切换到本分支并激活 lingbotvla，所有示例中的路径替换为真实路径，GPU仅使用分配给自己的编号。

先测现有baseline，1个任务、clean、1条episode：

```bash
python ysj_utils/evaluate.py --mode smoke --checkpoint /path/to/baseline/checkpoints/global_step_5000/hf_ckpt --gpus 0
```

可以附加 `--dry-run` 查看命令和协议；dry-run不加载模型、不验证环境，不算评测通过。

准备本轮成对实验。实际训练 robot config 必须来自原训练配置所指向的 clean 目录；如果字段布局不同会报错，需先核对。

```bash
python ysj_utils/action_experiment.py prepare \
  --robot-config /path/to/robot_config_clean/robotwin.yaml \
  --output /path/to/my_action_pair
```

默认使用 `xsy_utils/configs/robotwin_clean_baseline.yaml` 和其50任务清单。其他服务器应通过 `--base-config`、`--train-list` 提供事先核对过路径的副本。输出目录必须不存在。生成目录带机器绝对路径，应在实际训练服务器生成，不能直接复制Windows预览配置到Linux运行。

两组分别重新计算统计，不复用绝对动作的统计给delta。以下按顺序运行，避免上游统计程序的临时文件冲突：

```bash
python ysj_utils/action_experiment.py norm --bundle /path/to/my_action_pair --group B0
python ysj_utils/action_experiment.py norm --bundle /path/to/my_action_pair --group JDelta
```

成功后生成 `norm_stats.json.contract.json`，记录统计/机器人配置/训练清单的哈希及动作表示、chunk、norm类型。推理会核验这些信息；文件缺失、变更或不匹配直接报错。运行目录和配置/统计文件须随checkpoint保留。当前不支持只拷贝hf_ckpt后丢弃训练配置。

先分别执行5步短训，输出到各自 `train-smoke`；确认loss/梯度有限、checkpoint可重载，再用该checkpoint跑一次上面的smoke评测：

```bash
CUDA_VISIBLE_DEVICES=0 python ysj_utils/action_experiment.py train --bundle /path/to/my_action_pair --group B0 --stage smoke
CUDA_VISIBLE_DEVICES=0 python ysj_utils/action_experiment.py train --bundle /path/to/my_action_pair --group JDelta --stage smoke
```

这一步BS1/5步仅排错，不作为正式结果。`--dry-run` 可检查启动命令，但仍要求准备好的统计文件。

短测通过且GPU已分配后分别正式训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ysj_utils/action_experiment.py train --bundle /path/to/my_action_pair --group B0 --stage full
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ysj_utils/action_experiment.py train --bundle /path/to/my_action_pair --group JDelta --stage full
```

每组训练完后，用各自 `train/checkpoints/global_step_10000/hf_ckpt` 完成同seed评测：

```bash
python ysj_utils/evaluate.py --mode full --checkpoint /path/to/my_action_pair/B0/train/checkpoints/global_step_10000/hf_ckpt --gpus 0,1,2,3,4,5,6,7 --seed 0
python ysj_utils/evaluate.py --mode full --checkpoint /path/to/my_action_pair/JDelta/train/checkpoints/global_step_10000/hf_ckpt --gpus 0,1,2,3,4,5,6,7 --seed 0
```

输出在 `external/RoboTwin/eval_result/ysj_<mode>_<id>/run_*/`。结果校验要求完整的任务/环境集合、无重复且每格25条；即使底层脚本退出码为0，缺结果也会失败。任务正常执行但没有成功，计为任务失败，不误报为程序故障。

```bash
python ysj_utils/compare_results.py --baseline-run /path/to/B0/run_xxx --delta-run /path/to/JDelta/run_xxx --output comparison.json
```

比较脚本检查协议和关键训练条件、模型起点、训练清单及代码commit，输出逐任务和clean/random均值差异。目前是单训练seed的描述性结果，不能从汇总数字声称显著提升；还需核对场景一致性、失败视频，并用额外训练/评测seed复核。

## 当前待完成

- 在团队Linux训练环境复核；本地CPU测试已全部通过。
- 真实训练数据审计、norm计算、GPU bf16短训、保存重载。
- baseline小规模仿真、两组正式训练与350条评测。
- 当前没有实际任务成功率数据，没有证明Joint Delta提升。

服务器SSH端口可达，但本机免密登录失败；需要可用登录方式、真实baseline checkpoint路径与分配的GPU才能继续运行。
