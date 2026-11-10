<!-- 用途：补充本批六组GT有效深度条款及事后分层审计，保持原规划与冻结产物不变。 -->
# PROJECT_PLAN 补充说明：2026-11-10

本补充按用户明确选择执行：保留冻结文件，另存补充审计与修订说明，不迁移冻结版本。本文件不是原实验运行前的规划快照；分层明细是从原证据重建的**事后补充审计**。

## 1. 原规划引用与适用优先级

- 原规划：[PROJECT_PLAN.md](PROJECT_PLAN.md)
- 原规划绝对路径：`/home/amax/Documents/code/pixel-perfect-sfm-main/PROJECT_PLAN.md`
- 原规划 SHA-256：`7a7bcf93523c637a3b6aa1d7916dcaffa22625d646f60f66d4e1356b64a7aff0`
- 冻结记录：[shared/candidates/20261109_204552_747619/run.json](outputs/joint_map/shared/candidates/20261109_204552_747619/run.json)

原规划、原审计NPZ、冻结记录、模型、掩膜、DSM、基线指标、评价结果与原集成记录均保留原字节。原 SHA-256 交接链继续按原文件核验；本补充不改写这些摘要或路径。

**冲突条款：仅对第2节指纹绑定的六组GT数据，原规划第9.2节第1项中“未达到渲染远平面的深度才有效”的附加条件，由本补充第2节替代。** 基础有效性之外的反投影、地图范围、DSM采样有效性、D/D0、主验收及分区规则继续适用原规划。其他批次不自动继承此深度例外。

## 2. 本批GT基础有效深度规则

用户确认的规则为：

```python
valid_depth = np.isfinite(depth) & (depth > 0)
```

本批不再按far额外过滤，不删除最大深度值。该规则不证明正深度必然对应真实地表，也不新增背景识别算法。后续仍使用GT位姿、固定内参和相机z-depth反投影，再按输入地图范围与DSM有效性处理。

用户确认与助手接受该规则的记录、图像/深度配对、完整路径、尺寸及指纹已保存在原 [reference_manifest.json](outputs/joint_map/evaluation/reference_manifest.json) 的 `depth_validity` 中。本补充引用该原记录，不改写它。

六组文件位于 `datasets/images/DJI_20251221132525_0004_V/`，照片为同名PNG，深度为同名NPY：

| 配对名 | PNG SHA-256 | NPY SHA-256 |
|---|---|---|
| 0_0 | `f7a044317264e201aa40ef88a2ea81538c721f150e21923b75f92e472222d641` | `c52337404d01aa5984120d952bceee93ccffce94c6b137f4c9b19c7e4e92a8cb` |
| 1_0 | `91ee0dc2943bc49e0f9e83913efbb55fe3f7f0915ad8210ef0df42191029c210` | `74200b3afdbbb0962bae85c57ac9b92bf268547003fa9e036dc4d1b189c5bdf6` |
| 2_0 | `f4d86b2b60246f19aba4ce2a1f9e938f92b32d154c83cca79c4da1e77eb1c6bc` | `32a46c9df23fb74ff85fc51b472740947f29a9f3f99eed85b9ae21790d65ae5c` |
| 3_0 | `8a8a46280cbac2923288197531be0441016a60b60cca162a94631d66ed93fae2` | `f401056dce91e53c1d092d0c2160551c4668682791205e6e7d9f9454f25acdf4` |
| 4_0 | `c6915443b510e904d094b0dc2c4414cd20452c98c2a8b1f420f6de58ef56c256` | `f660556e17b18a39010720f39ea55f9e443c6b09b9a2d739007c0c1677e3f26f` |
| 5_0 | `6da0a893dc57e4abf17b00010c790e850b2852a99197ad4c3fc7625cfad8c0c4` | `08b9bf7aa7fd9eca136242dcb02d78fc1567a430dcbcf3b7a2eab234a55e66d8` |

## 3. 事后分层审计范围与规则

原始来源是两分支 `<branch>/map/point_evidence.npz`：

| 分支 | 原NPZ SHA-256 |
|---|---|
| baseline | `6ff824e06c2566ca5e4e0b1f728df3d27b1d9abb9f53f057efd590e1176880bd` |
| joint | `c5e6eb392eabf94452a0d5fe5b26de64b57f1b303a5c46e8343068d0dad7d0bb` |

只取原NPZ已经标记为 `reliable & inside_raster` 的不同稠密点，使用原像元、原float64椭球高和原点ID。此次不重新计算点级可靠性、不生成匹配、不重建模型、不改变遮挡或稀疏冲突判据，不重新评分GT。

对每个原像元按高度稳定排序，相邻高度差≥1.0 m分组；组内至少3个不同稠密点才构成有支持层。两个及以上有支持层时，主高度拒绝原因为 `multiple_height_layers`；否则存在多个分离组时为 `ambiguous_height_support`，小组完整保留。单组继续按原最少3点、未缩放MAD≤0.5 m检查。没有合格点的像元保留 `not_evaluated`。

`multiple_height_layers` 的优先级高于 `ambiguous_height_support`。1.0 m仍是本批固定工程阈值，不宣称识别所有重叠或连续连接的多层表面。

审计逐像元核对原主高度原因、可靠点总数、median和未缩放MAD。主高度原因与稀疏—稠密冲突分开解释：原 `sparse_conflict_mask`、四掩膜和实际更新结果不重写。出现不一致时，在补充JSON中保存原值、重建值和像元位置，不能据此替换原记录。

## 4. 新增文件与读取方式

所有程序、测试及审计产物均放在 [acceptance_audit_20261110](outputs/joint_map/acceptance_audit_20261110/)：

| 文件 | 含义 |
|---|---|
| `build_height_audit.py` | 本批独立补充程序，不修改现有源码 |
| `test_height_audit.py` | 分组边界、身份、优先级及差异保留测试 |
| `baseline_height_groups.npz`、`joint_height_groups.npz` | 全像元分组明细与原/重建主高度原因 |
| `baseline_height_groups.json`、`joint_height_groups.json` | 来源、输出摘要、统计、逐像元不一致记录 |
| `preserved_before.json`、`preserved_after.json` | 原文件执行前后大小与SHA-256快照 |
| `pipeline_readonly_audit.json` | 直接调用原 `audit_pipeline(...)` 的只读结果 |
| `manifest.json` | 事后标记、来源与输出摘要、原文件变化及总核验结论 |

NPZ字段：

- `raster_shape`：原网格形状；行列和索引均从0开始。
- `cell_group_offsets[H*W+1]`：像元 `c=row*W+column` 的组范围为 `[offsets[c], offsets[c+1])`，空像元范围为空。
- `group_counts`：每组不同稠密点数。
- `group_min_height_m`、`group_max_height_m`：每组高度范围，float64、椭球高米制。
- `group_gap_to_next_m`、`group_has_next`：下一组最小高度减本组最大高度；末组为NaN且has_next=false，表示不适用，不用0伪造间隔。
- `group_has_supported_layer`：点数是否≥3。
- `group_point_offsets`、`group_point_ids`：各组的原稠密点ID，用于核对不同点数及范围。
- `primary_height_reason_original`、`primary_height_reason_reconstructed`、`height_reason_names`：原与重建的主高度原因，包含未评估像元。
- `reliable_point_counts_reconstructed`、`height_medians_reconstructed`、`height_mads_reconstructed`：与原记录逐项比较的统计。

## 5. 运行与核验边界

仅在项目根目录、现有pixsfm解释器中执行，不修改环境：

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -m unittest discover -s outputs/joint_map/acceptance_audit_20261110 -p 'test_height_audit.py' -v
/opt/anaconda3/envs/pixsfm/bin/python -B -m unittest discover -s tests -p 'test_joint_map.py' -v
/opt/anaconda3/envs/pixsfm/bin/python -B -m outputs.joint_map.acceptance_audit_20261110.build_height_audit
```

最后一条为首次生成命令。审计文件采用独占新建，已有补充不被覆盖。生成后检查 `manifest.json` 和每分支JSON；存在任何不一致时，结果必须显示为不一致而非改写原记录。

执行前核对原集成摘要和13项模块3冻结摘要；生成后再次核对冻结摘要，直接调用只读 `audit_pipeline(config_path,evaluation_config_path,output_dir)`，并比较旧源码、旧规划和全部旧流水线产物的前后快照。不会调用会写旧manifest的 `--stage all`。

补充审计闭合的是分层明细交付与规划条款一致性缺口。原八项主验收中旋转median/std、DSM median/std未通过的结论仍保留，不能把本补充通过解释为精度改善或整体交付完成。
