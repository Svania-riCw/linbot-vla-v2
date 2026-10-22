# 全图432 + 观测图像增强

## 实验目的与默认输入

检验：固定三路384全图、432视觉patch tokens和训练计算条件，只增加训练时的观测图像扰动，能否提高random场景成功率，同时保持clean性能。

`roi` 分支的两个标准 RoboTwin YAML（`robotwin.yaml`、`robotwin_dist_muon.yaml`）现在显式使用 **三路全图384、Lanczos4、固定processor网格、ROI关闭**，学生网格为每路12×12。标准配置本身保持无增强，作为输入表示默认值；增强实验由下面的生成器显式启用。

这两个文件仍是上游的通用训练模板，不是本实验的global128/10k完整启动配方。不要直接拿模板启动本实验；生成器必须读取已经验证过的正式训练YAML。旧ROI代码、已生成的旧实验YAML和checkpoint不被改写；显式传入ROI配置仍可复现旧输入。

## 按参考图实现的部分

每个训练样本先抽一次门控：10%原样通过，90%进入增强。下表概率均是**进入90%分支之后**的条件概率，不是全体样本发生率。

| 操作 | 条件概率 | 参数 |
|---|---:|---|
| 颜色/色温 | 80% | 亮度0.25–1.10、对比度0.5–1.5、饱和度0.4–1.6 |
| 二维缩放/平移 | 50% | 中心缩放0.90–1.10；x/y分别均匀平移±6%宽/高，露出区域补黑 |
| 矩形遮挡 | 95% | 仅在有对齐前景mask且找到安全矩形时执行；纯色/马赛克/像素噪声等概率 |
| 高斯噪声 | 50% | 与椒盐独立抽样，可以同时出现 |
| 椒盐噪声 | 15% | 单个像素RGB一起置黑或白 |
| 高斯模糊 | 25% | 与运动模糊互斥 |
| 运动模糊 | 10% | 另65%不做模糊 |
| JPEG | 30% | 整数质量35–90，RGB/BGR显式转换 |

固定执行顺序：颜色→缩放/平移→遮挡→高斯噪声→椒盐→模糊→JPEG；后续噪声/压缩可能改变原先的黑边。增强分支也可能没有操作被抽中，因此最终像素未变的样本比例可以略大于10%。不会强制重抽来改变概率。

不换背景/材质，不粘贴真实物体，不重建几何/重布灯光，不用生成模型，不改任务文本、关节状态或动作标签。二维仿射只改变RGB观测，不改变模拟器场景；它仍可能把边缘物体裁出画面，不能据此宣称所有样本语义信息完整保留。

## 参考图没有指定的实现选择

以下是本次可配置的保守初值，**不是来源配方已经公布的参数**，不声称逐项复现原作者实现：

- 色温：`t ~ U[-0.10,0.10]`，RGB乘以`[exp(t),1,exp(-t)]`。这是通道增益近似，不是开尔文色温标定。
- 遮挡：每视图至多1个矩形，面积占2%–10%，宽高比按log均匀0.5–2.0；最多尝试20次；前景外扩4像素。马赛克按约8像素块降采样再最近邻放大。
- 高斯噪声标准差：0.005–0.030（相对于[0,1]）；椒盐像素比例0.001–0.010。
- 高斯模糊sigma：0.10–1.50；运动核长度3/5/7等概率，角度0–π均匀。
- 颜色、几何、门控、模糊和JPEG参数在同一样本三相机/当前未来帧间共用；各相机的噪声纹理和遮挡位置独立。每相机当前/未来帧复用噪声纹理、矩形位置，避免额外随机闪烁。这里没有宣称不同相机对应同一个物理相机变换。

配置集中在 `configs/vla/robotwin/full432_augmentation_recipe.yaml`；解析器拒绝未知字段和不合法概率/范围，防止拼错参数后悄悄失效。

## 前景mask接口（用户确认：缺失时跳过遮挡）

没有绑定或猜测检测器。提供者应在 `FeatureTransform.apply(raw_item)` **之前**加入：

```python
raw_item["augmentation_foreground_masks"] = {
    "observation.images.cam_high": head_masks,
    "observation.images.cam_left_wrist": left_masks,
    "observation.images.cam_right_wrist": right_masks,
}
```

每项是numpy数组或CPU tensor。单图为`[H,W]`；当前/未来两帧为`[2,H,W]`，必须和raw_item选出的帧顺序一致，不能把整段episode或只当前一帧的mask冒充两帧。图像已经经过同一384缩放，mask也必须对齐该分辨率（提供者应使用保守的mask缩放方式）。接受bool、0/1或0/255；非零表示需要保护的机器人/物体，0表示已知背景。**全零mask表示确实检测到全背景，不是“检测失败”。**检测失败应省略该相机mask或使用全一保护。

mask随RGB使用同一个二维仿射变换，然后取所有选中帧前景并集并外扩，矩形与该区域必须零重叠。mask仅用于训练扰动避让，不进入策略输入；并集使用未来mask仅保护教师目标，不改变原图ROI或动作标签。无安全位置时跳过，不扩大尝试次数/放宽前景保护。

当前数据没有mask，故实际运行是**除了矩形遮挡以外的增强组合**；不能称“95%成功进行了前景保护遮挡”。`last_augmentation_report`按视图区分`not_requested`、`applied`、`skipped_missing_mask`、`skipped_no_safe_rectangle`，用于小批次检查。可设置`missing_foreground: error`用于mask准备后的严格核验。错位/非二值mask直接报错，数据集不会将其替换为随机样本。

## 训练/教师/评测与恢复

新增强在原始三路图像已缩到384之后、特征映射和归一化之前执行。当前/未来帧一起处理，因此视觉输入和深度/DINO教师看到相同增强；旧`prepare_images`颜色抖动不会再叠加一次。教师网络和loss权重均不改，也不把干净目标和增强观测的几何位置混用。这个选择本身需要实验验证，不能预先声称教师伪标签质量会提高。

`policy_eval=True`完全绕过增强且不消耗增强随机数；推理YAML导出也显式写入`image_augment: false`。HF模型配置记录解析后的增强配方hash，DCP恢复核验一致，防止中途增删增强或更改强度。旧无增强checkpoint与旧ROI推理兼容。

每样本随机种子来自DataLoader worker的torch RNG，内部局部NumPy Generator不改全局NumPy状态。固定seed可重放同一变换；不承诺改变worker数或预取状态后训练逐位复现。

## 生成两份可审查的配方（不会启动训练）

```bash
python tools/prepare_full432_augmentation.py \
  --base /path/to/verified/lingbotvla_cli.yaml \
  --recipe configs/vla/robotwin/full432_augmentation_recipe.yaml \
  --destination /path/to/new_aug_configs \
  --run-root /path/to/new_aug_runs
```

生成`full432_control.yaml`和`full432_aug.yaml`。可读取已完成ROI运行的原始训练YAML，但会在两者中都清除ROI/几何字段、旧resume路径，保留原始6B初始化；不能读取以训练checkpoint为model_path的配方。模型/优化器/LR/seed42/compile/教师、clean数据/norm路径、global128=micro4×8×accum4、FP32/混合精度/action精度设置、workers8/prefetch4/pin_memory全部保持。生成器对锁定值不符报错，不替你悄悄改参。

两份新配方均保持max_steps/scheduler10000、每1000保存、5000暂停，恢复时仍从本次5k完整DCP继续。配置生成不是远端输入/clean norm/空闲资源核验，也不是自动提交；不重训既有baseline。历史full384的早期worker轨迹与新配方仍有差异，报告对照时要保留该限制。

下一步先核验真实训练样本预览和短程GPU保存/恢复，再决定完整10k。5k/10k按相同7任务×clean/random×25评测；报告成功次数、百分点差，以及实际执行/跳过的遮挡比例。目前只完成代码和CPU验证，**没有新成功率或GPU速度结果**。

## 本地验证

真实RGB文件可用同一实现输出预览和抽样参数（不使用合成模型，也不启动训练）：

```bash
python tools/preview_observation_augmentation.py \
  --images /path/to/head.png /path/to/left.png /path/to/right.png \
  --output /path/to/new_preview --seeds 0 1 2 3 4
```

该预览没有mask，因此明确标记未执行前景保护遮挡。测试中的人工mask只验证避让逻辑，不充当真实检测结果。

```bash
python -m unittest discover -s tests -p test_observation_augmentation.py -v
python -m unittest discover -s tests -p test_roi432.py -v
```

新测试包括条件概率、模糊互斥、RGB顺序、mask避让/错位拒绝、当前未来一致、10%逐像素无改动、真实Qwen 432 tokens、状态/动作不变、评测绕过、配方hash恢复保护与训练参数解析。真实Qwen测试使用本地processor元数据；缺失时显式skip，不冒充实际验证。
