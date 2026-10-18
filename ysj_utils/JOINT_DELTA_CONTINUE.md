# Joint Delta 5k → 累计10k续训

后续compile=False复测已完成350次，退出码0，耗时1小时00分35秒：Clean97/175=55.43%、Random39/175=22.29%、总计136/350=38.86%。相同10k checkpoint的compile=True结果为127/350=36.29%，本次多成功9次；对比原5k的compile=False结果124/350=35.43%，本次多成功12次。原5k Clean为52.00%，本次55.43%，此前Clean下降的现象在本次复测未复现。checkpoint、归一化、机器人配置、训练配置、任务/环境/次数、环境seed0、推理fp32及chunk50已逐项核对一致；策略内部随机噪声未按episode配对，不能仅据一次复测断言所有差异由compile导致。新报告为`reports/h20_joint_delta_10k_compile_false_2026-10-18.json`，服务器运行目录为`/2025233210/ysj/lingbot_ws/outputs/joint_delta_5k_to_10k/reeval_compile_false_20261018`。

最终结果（10月18日核验）：续训到累计10000步及compile=True的350次评测均完成，退出码0；Clean86/175=49.14%、Random41/175=23.43%、合计127/350=36.29%。本轮评测57分18秒。原5k为124/350=35.43%，本轮多成功3次；推理compile开关和学习率历史不同，不能单独归因于训练步数。历史10k baseline162/350=46.29%，本轮低10个百分点，未获得整体涨分。完整协议与结果记录为`reports/h20_joint_delta_10k_compile_true_2026-10-18.json`。

5k与续训10k的实际配置逐项核对记录为`reports/joint_delta_10k_detail/configuration_audit_5k_vs_10k.json`：train-list、norm、robot config、EE状态特征及标签清单SHA256均相同；FP32、mask、batch、seed及模型配置保持一致。实际续训日志恰好包含5001至10000共5000次更新，无从头训练或跳过优化器的日志。另有prefetch4→1、恢复/输出/总步数参数以及显式保留5000步LR horizon的差异。评测使用相同任务、episode数、环境seed0、fp32和chunk50，但compile由False改True。模型服务启动时固定随机种子，episode reset未重置策略随机数，且8卡动态领取任务；因此没有验证逐episode相同策略噪声，不能将现有结果当作严格配对比较。Clean少5次成功不构成过拟合的直接证据。

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

H20真实恢复检查发现该容器`/dev/shm`只有4GB且不能重挂载扩容，保留原预取配置会在恢复后首步编译期间导致DataLoader worker Bus error。逐项核对实际保存的CLI配置后确认，原`data.prefetch_factor=4`，续训为1；先前“2→1”的表述使用了函数默认值，现予纠正。保留8个worker和加载器恢复状态；它减少在途batch，不改变batch大小、数据集或优化目标。这个资源配置差异也记录在continuation.json中，可用`--prefetch-factor 4`保留原值，但需先解决共享内存容量。

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

## 续训验证

H20环境通过96项回归检查，报告 `reports/h20_resume_checks_2026-10-17.json`。真实8卡检查从原DCP5000完整恢复到5001、5002：loss分别0.0241、0.0265，GradNorm分别0.3514、0.3898，主组LR均5e-5；已保存新DCP和完整HF，进程退出码0。归一化和机器人配置SHA256与原模型一致。记录为 `reports/h20_resume_gpu_2026-10-17.json`，诊断产物位于 `/2025233210/ysj/lingbot_ws/outputs/joint_delta_resume_check_5002`。

该检查验证了恢复状态、两次更新及保存；它本身没有执行完整额外5000步或compile=True的仿真rollout。保留原prefetch=4的检查因4GB共享内存不足退出，日志仍保留在当时误命名的`joint_delta_resume_check_5002_failed_prefetch2`，没有覆盖原5k模型。

正式续训已在30375的`ysj-jdelta10k`会话启动，输出目录为本文默认目录；已确认从5000恢复并完成5001及后续多次更新，loss/梯度有限。检查快照为`reports/h20_joint_delta_continue_started_2026-10-17.json`，它只证明启动时运行正常，不代表后续10k训练和350次评测已完成。查看使用`tmux attach -t ysj-jdelta10k`，默认选中logs窗口；断开后重连即可，只有进程确实退出才重复启动脚本。
