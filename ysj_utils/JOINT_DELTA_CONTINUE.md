# Joint Delta 5k → 累计10k续训

入口：`bash ysj_utils/resume_joint_delta.sh`。建议在30375上自己的tmux窗口执行。默认不会修改旧训练目录；新结果写到 `/2025233210/ysj/lingbot_ws/outputs/joint_delta_5k_to_10k`。

先检查完整DCP和命令，不启动GPU：

```bash
cd /2025233210/ysj/lingbot_ws/linbot-vla-v2
bash ysj_utils/resume_joint_delta.sh --dry-run
```

正式运行：

```bash
tmux new -s ysj-jdelta10k
cd /2025233210/ysj/lingbot_ws/linbot-vla-v2
bash ysj_utils/resume_joint_delta.sh
```

断网后重连tmux即可，离开窗口用Ctrl+B再D。进程确实停止时，重复同一脚本：核对所有8个rank、模型/优化器DCP元数据及分片大小，选择新目录最新完整checkpoint；未保存的步数会重做。完整恢复优化器、scheduler、加载器进度和原实现保存的CPU torch RNG；并不承诺CUDA/Python/NumPy随机状态的逐位复现，因为上游没有全部保存它们。

恢复失败会报错退出，禁止静默从头训练或跳过优化器。原模型、归一化统计、EE标签均不覆盖；新目录有进程锁和源文件指纹。每1000步保存一次；训练结束后执行7任务 × clean/random ×25 =350次评测，启用`--use-compile true`，推理仍为fp32、chunk50、seed0。结果为新目录的`evaluation_compile_true.json`，训练/评测日志分别为`train_<UTC时间>.log`、`eval_<UTC时间>.log`。遇到未完整结束的评测不会自动续成成功报告。

## 保留什么、改变什么

本轮从现有Joint Delta的完整5k DCP继续，新增5000次参数更新，到全局10000步停止。保留原`enable_fp32=False`、`enable_mixed_precision=True`、`mask_action_padding=True`、BS128、8卡micro16、seed42、chunk50、数据列表、EE输入、norm、优化器、训练compile及所有模型配置。新输出目录、总步数和恢复参数会变化。

学习率必须单独说明：原5k余弦已降至主组5e-5，Muon专家组约1.4142e-4。本脚本将scheduler原始horizon固定为5000，后续维持其末端值，避免仅改max_steps=10000导致LR回升。因此这是“原5k计划之后续训5k”，不等于从第1步按10k余弦训练；比较历史10k baseline时需注明。

## 与原absolute-joint baseline相比，已有的其他差异

- Joint Delta：12个关节未来目标减去当前观测关节值；夹爪保留绝对目标。推理按同一锚点还原，继续执行joint控制。
- 当前输入同时包含joint、夹爪和双臂EE位姿；即使JDelta不预测EE，EE状态仍进入输入。原baseline未包含EE状态。
- 针对delta重新计算归一化，训练和推理使用同一stats；读取侧边EE标签并校验时序。
- Joint+EE额外预测EE局部增量，并有`lambda=0.1`的分组辅助损失。JDelta没有该辅助监督。
- 实现还修复过数据迭代结束时虚增global_step、统计入口rank环境，以及减少无关图片解码/CPU过量线程。读取投影做过数值等价性核对，但代码实现与历史baseline提交不同。
- 当前没有启用通用维度/时间加权、Spline、RTC或VQA等额外方法。原始模型、55维flow head、初始化、Muon、基础LR、depth/video辅助监督保持原方案。
- 既有padding/精度差异继续保留；`enable_resume`只是新一轮恢复控制。新增配置里出现但未启用的weighted/spline字段不应算成实际训练了这些方法。

## 既有结果

两组350次评测均完成，退出码0：JDelta5k clean52.00%、random18.86%、合计124/350=35.43%；Joint+EE10k clean56.00%、random24.57%、合计141/350=40.29%。评测分别65.7、55.2分钟。原两次推理compile=False，新的compile=True结果需分开记录，不能混称完全同一评测条件。
