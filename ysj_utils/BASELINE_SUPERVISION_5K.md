# 与 baseline 时间补齐方式对齐的 Weighted / Spline 5k 实验

用户于2026-10-18明确选择重开：**保留原动作维度 mask，取消额外时间 padding 屏蔽，补齐的末尾重复动作参与训练。** 两组均从官方预训练模型重新开始，各5000次更新，不从旧任务中途权重续训。

| 项目 | 两组公共配置 |
| --- | --- |
| 配置来源 | 团队实际 baseline_gbs128_5k/lingbotvla_cli.yaml，记录main及实际代码提交 |
| 动作和输入 | baseline绝对12维关节＋2维夹爪，无额外EE state，无Joint Delta |
| 数据和统计 | 原50个clean task，复用baseline绝对动作归一化并校验SHA256 |
| 训练 | BS128，8卡每卡16，5000 updates，seed42，原Muon和5k cosine学习率 |
| 精度 | enable_fp32=True、enable_mixed_precision=True，与baseline相同；冻结视觉教师保留原bf16 autocast |
| mask | 动作维度mask保留；mask_action_padding=False，末尾补齐位置参与监督 |
| 其他模型配置 | 保留L1_fm、深度/视频辅助目标、图像配置、训练compile=True |
| 资源适配 | prefetch_factor从4降为1，worker仍为8，适配容器4GiB共享内存 |
| 评测 | 各350次：7任务×clean/random×25；seed0、fp32、compile=False |

Weighted依据：[Weighted Conditional Flow Matching](https://arxiv.org/abs/2507.22270)第3节与附录D。距离计算、Z校准、动作loss全部包含50步的真实动作维度，包括末尾重复位置。从50任务各抽8段（400段），配合128组独立Gaussian噪声，重新扫描带宽，选择两个边际倾斜代理相对方差均≤0.01的最小候选值，并估计固定log_Z。使用同一次forward的噪声，不在每张卡minibatch内重算归一化。原论文主要使用平方误差与无条件生成；本次保留baseline的L1，因此是其权重思想的适配，不保证机器人成功率提升。

Spline主要依据：[BEAST，NeurIPS 2025主会](https://papers.nips.cc/paper_files/paper/2025/hash/fc75cfcec16170f2b54d00283c739928-Abstract-Conference.html)，[论文第4节和附录D](https://arxiv.org/html/2506.06072v3)。将完整50步归一化绝对动作（包括重复末尾动作）拟合为16个三次clamped B-spline控制点。原action head预测控制点，使用原L1 flow loss；推理先还原50步，再按原统计反归一化并交给qpos控制器。此次不再只拟合真实前缀，所有有实际动作维度的控制点参与监督。保留旧的padding-aware模式以兼容历史checkpoint，但本次明确关闭。

本次不离散化token、不更换VLM、不加入跨chunk边界约束，因此不宣称完整复现BEAST或保证跨chunk连续。准备阶段记录完整50步与真实时间步的重建误差，检查平滑是否损失接触或夹爪精度。

补充阅读：[B-spline Policy，2026预印本](https://arxiv.org/abs/2607.09648)，不冒充正式会议论文。[Neural Dynamic Policies，NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/354ac345fd8c6d7ef634d9a8e3d47b83-Abstract.html)提供轨迹参数化动机，但使用DMP，不是本次算法。

## 验证和运行

本机108项测试通过。新增验证覆盖：Gibbs均匀权重极限与实际baseline loss一致；补齐位置参与梯度、无效维度仍不参与；完整50步校准不受padding标记影响；Spline实际policy拟合完整50步；收据绑定时间补齐策略并兼容旧收据。

```
python -m unittest discover -s tests/action_parameterization -v
python ysj_utils/run_baseline_supervision.py --run-dir /path/to/new_run --prepare-only
python ysj_utils/run_baseline_supervision.py --run-dir /path/to/new_run --run-prepared
```

准备产生source_baseline.yaml、plan.json、gibbs_calibration.json、spline_fit.json及Control/Weighted/Spline独立配置。Control仅生成配置，不追加第三组5k训练。两组enable_resume=False、load_checkpoint_path=None；启动目录全新，不续训旧任务。

队列：**Weighted 5k → 350次评测 → Spline 5k → 350次评测**。异常或非有限loss/梯度会停止队列并记录status；每1000步保存DCP/HF。进度解析已适配日志的epoch分母4295，优化器预算仍为5000。

新队列已于2026-10-18 21:54（北京时间）在30375的8张H20启动，代码提交c5aa3cb。本机及服务器均108项测试通过；已确认Weighted完成最初多次真实更新，梯度与学习率正常，Spline尚在队列中。新输出根目录：`/2025233210/ysj/lingbot_ws/outputs/baseline_supervision_5k_aligned_20261018`。

重新校准得到epsilon=0.53452266、log_Z=-2.23628366，源/目标相对方差约0.00225/0.00907；400段共20000个时间位置全部参与校准和样条拟合，其中15849个为原始真实时间位置。新旧epsilon数值相同是网格选择结果，但log_Z已重新计算，监督范围已改变。

查看日志：`tmux attach -t ysj-supervision`。启动证据与实际配置审计：[reports/baseline_supervision_5k_aligned_launch_2026-10-18.json](reports/baseline_supervision_5k_aligned_launch_2026-10-18.json)。按起跑后约4.25秒/步估计，Weighted约6小时；两组训练和两轮评测暂估12–15小时，后续吞吐和导出会影响耗时。

## 被替代的旧尝试

旧输出：/2025233210/ysj/lingbot_ws/outputs/baseline_supervision_5k_20261018。旧Weighted约433步时按用户要求停止，旧Spline未启动。日志保留，状态SUPERSEDED，会话ysj-supervision-masked-stopped。旧尝试使用时间mask，不能混入新5k实验或拼接预算。

旧启动证据：[reports/baseline_supervision_5k_launch_2026-10-18.json](reports/baseline_supervision_5k_launch_2026-10-18.json)，仅作历史记录。旧文档可在git提交e3c9109查看。

## 比较限制

新实验消除了额外时间mask的混杂因素。仍记录prefetch4→1资源适配、代码版本、实际配置、统计和评测条件；不能仅凭同seed宣称逐episode策略噪声一致。不同监督目标的训练loss不能直接横比，最终比较同预算5k checkpoint的350次成功率及逐任务数据。每任务每环境25次，1次成功=4个百分点，小幅变化需要多seed复测。
