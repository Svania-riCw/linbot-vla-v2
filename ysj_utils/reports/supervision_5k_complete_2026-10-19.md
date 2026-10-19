# Weighted / Spline 5k 完整结果

两组均5000次更新、BS128、fp32，各350次评测，compile=False。保留动作维度mask，末尾重复补齐动作参与监督。

| 方法 | Clean | Random | 总体 |
| --- | --- | --- | --- |
| Baseline5k | 77/175 (44.00%) | 47/175 (26.86%) | 124/350 (35.43%) |
| Weighted | 66/175 (37.71%) | 28/175 (16.00%) | 94/350 (26.86%) |
| Spline | 87/175 (49.71%) | 34/175 (19.43%) | 121/350 (34.57%) |

| 任务 | Baseline clean/random | Weighted clean/random | Spline clean/random |
| --- | --- | --- | --- |
| blocks_ranking_rgb | 9/25 · 8/25 | 12/25 · 4/25 | 18/25 · 7/25 |
| hanging_mug | 7/25 · 1/25 | 4/25 · 0/25 | 3/25 · 1/25 |
| open_laptop | 21/25 · 12/25 | 21/25 · 8/25 | 20/25 · 7/25 |
| pick_diverse_bottles | 8/25 · 3/25 | 2/25 · 2/25 | 0/25 · 2/25 |
| rotate_qrcode | 13/25 · 6/25 | 11/25 · 5/25 | 15/25 · 2/25 |
| stack_blocks_three | 2/25 · 3/25 | 1/25 · 1/25 | 11/25 · 3/25 |
| place_shoe | 17/25 · 14/25 | 15/25 · 8/25 | 20/25 · 12/25 |

Spline clean比历史baseline多10次成功，random少13次，总体少3次；Weighted总体少30次。单次训练与有限评测不能证明稳定提升或退化，也不能确定失败原因。

这是与历史baseline的描述性比较；原checkpoint尚未在当前代码/环境重评，未验证逐episode策略噪声一致。预取4→1、代码版本等差异已记录。

训练与checkpoint导出、两轮评测均正常结束；不能把训练器内部自动评测hook的eval_failed计数误读为本次外部350次评测失败。
