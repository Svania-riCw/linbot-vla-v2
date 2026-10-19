# Spline 5k → 10k 真实续训

用户要求在已有Spline训练基础上新增5000次更新，再评测。源模型是完成350次评测的Spline 5k：clean87/175（49.71%）、random34/175（19.43%）、总体121/350（34.57%）。

- 源目录：`/2025233210/ysj/lingbot_ws/outputs/baseline_supervision_5k_aligned_20261018/Spline/train`
- 新输出：`/2025233210/ysj/lingbot_ws/outputs/spline_5k_to_10k_20261019`
- 从源`checkpoints/global_step_5000`的完整DCP恢复模型、优化器和已保存的scheduler/dataloader/随机状态；不从仅含权重的HF目录当作完整训练恢复。
- 验证8个rank的extra_state、模型与优化器DCP元数据及分片大小。严格恢复失败即停止，不能退回重新训练。
- 保留16个三次Spline控制点、绝对关节动作、原动作维度mask、`mask_action_padding=False`、fp32、BS128、8卡micro16、seed42、归一化统计、数据列表、prefetch1、原训练compile设置及全部辅助目标。
- 原5k cosine已到最低学习率。续训显式保留5000步scheduler horizon，主学习率继续5e-5（专家组保持其原倍率），不重启衰减，不把max_steps改成10000后重新解释前半程。
- 目标为global step10000，即真实新增5001—10000共5000次更新。每1000步保存DCP/HF，输出与源目录分离。
- 完成后自动按7任务×clean/random×25=350次评测，fp32、seed0、`compile=False`；结果写`evaluation_compile_false.json`。

```bash
python ysj_utils/resume_spline.py --dry-run
bash ysj_utils/resume_spline.sh
```

在tmux中执行。重复同一命令时，脚本选择新输出目录里最新完整的DCP继续；损坏的中间断点被记录，参数/源文件/数据合同变化则拒绝混用输出目录。

2026-10-19 13:52（北京时间）在30375的8张H20启动，会话`ysj-spline10k`，代码提交`065ac4e`。已确认完整恢复global_step_5000，首次真实更新为5001、5002、5003，loss分别0.0241、0.0322、0.0287，梯度范数分别0.5062、0.4943、0.4981，主学习率均5e-5；采集证据时已到5005。本机10项恢复专项测试、服务器全部110项测试通过。

对比实际保存的源配置与新配置，data/model完全一致；train仅变化6项：output_dir、max_steps、enable_resume、load_checkpoint_path、require_resume、显式lr_schedule_steps=5000。模型、动作、mask、精度和优化器超参未增加变化。

验证证据：[reports/spline_10k_continuation_started_2026-10-19.json](reports/spline_10k_continuation_started_2026-10-19.json)。查看日志：`tmux attach -t ysj-spline10k`。训练日志与最终评测结果均写入上述独立输出目录。

这是**5k终点继续训练到10k**，不是从一开始使用10k cosine的独立10k实验。评测首先与原Spline 5k比较；与历史baseline10k比较时必须同时说明学习率路径、代码版本和评测随机性差异，不能仅凭总步数相同宣称完全同预算同配置。
