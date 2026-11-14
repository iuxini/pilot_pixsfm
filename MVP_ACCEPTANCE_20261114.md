<!-- 用途：记录按PROJECT_PLAN.md执行的P4 MVP验收、真实逐阶段证据、硬编码审查与使用设置。 -->
# P4 · MVP整体验收报告（2026-11-14）

> 后续交付入口：用户新增“仅提供六张照片、DOM/DSM、输出路径和内参，每次全新生成”的要求由`run_localization.py`实现；当前用法见[USAGE_JOINT_MAP.md](USAGE_JOINT_MAP.md)。本文保留当时采用已有原版产物、手动确认及GT评价的验收证据和精度结论，逐阶段命令不再是日常定位的必需操作。

**结论：工程功能与约定输入下的自动化验收通过；按计划第9.3节的总体验收未通过。** 八项精度仅4项达标，`primary_acceptance_passed=false`、`delivery_complete=false`。因此本报告不签发无条件的“【MVP验收通过】”。

验收依据：[PROJECT_PLAN.md](/home/amax/Documents/code/pixel-perfect-sfm-main/PROJECT_PLAN.md)第4、5.3、5.4、6、9及10节，以及本批明确确认的有限正深度规则。计划中旧实验/未开发状态是冻结时的历史说明，不回写摘要；本次状态以实测日志和本报告为准。

项目根目录：`/home/amax/Documents/code/pixel-perfect-sfm-main`。本轮独立验收目录：`/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114`。原实验重建/地图/已冻结GT基线及失败评价保留；本次不调整算法、权重选择策略、可靠性阈值、环境或原始数据。

## 第一步：逐条对照必须需求

| 必做需求 | 实现状态 | 核验内容与范围 |
|---|---|---|
| 环境和输入核验 | ✅ 已实现 | 指定pixsfm解释器、已有依赖、本批输入指纹、RGB/K、地图格式和原版来源核验。 |
| WGS84/ECEF/ENU及姿态/像元坐标 | ✅ 已实现 | 固定地图原点，CV相机轴、椭球高、像元中心与双线性采样；回归覆盖非零姿态。 |
| 原版对照与初值 | ✅ 已实现 | 复用本批原版SfM/MVS；只作共同地图Sim(3)，保留原版相对几何；不把外部配准称为原生WGS84输出。 |
| DOM匹配与确定track身份 | ✅ 已实现 | SuperPoint/SuperGlue、变体坐标还原、KA特征行号/双向tracks身份、3D独立验证。 |
| 共同一致性筛选 | ✅ 已实现 | 保持2 m；原始候选及全部图像tracks保留，超限关联标拒绝，仅通过者重配准并复查；数量动态。 |
| 动态划分及冻结训练地图集合 | ✅ 已实现 | 合格tracks按0.2/seed=0留一观测；训练/留出不重叠且覆盖完整；冻结后严格核对ID、均值、协方差和摘要。 |
| 整幅特征缓存自动准备 | ✅ 已实现 | 本轮从空目录实际提取六图；缺失自动生成、存在按图像/权重/配置/内容核验，拒绝同名异内容。 |
| 原生绑定与数值校验 | ✅ 已实现 | 残差维数、安全求值、原生manifold、同状态代价/扰动、多尺度128通道公共窗口与7自由度地图信息检查。 |
| 三候选验证及显式确认 | ✅ 已实现 | 0.1/1/10完整验证，本轮均passed；未确认禁止register/joint；同批授权重放有逐值一致性证据。 |
| 正式联合优化 | ✅ 已实现 | 固定K，六pose与共享XYZ可变；图像/几何/地图三类因子同一XYZ；300/100预算、域保护/恢复与实际因子审计。 |
| 两分支稠密重建 | ✅ 已实现 | 原版米制转换；联合模型实际CUDA MVS；有效深度/法线、点序、可见性和相机/尺度一致性。 |
| 可靠性和局部DSM更新 | ✅ 已实现 | 真实可见/重叠、3视图、视差、重投影、深度/法线、像元分层/MAD；四掩膜、局部冲突及未更新原值/NoData保留。 |
| 稀疏—稠密—像元证据链 | ✅ 已实现 | 原地图/初值/优化XYZ、支持视图、残差、可靠稠密ID与更新像元offsets；无可靠支持不强写。 |
| WGS84导出 | ✅ 已实现 | 两分支各六行位姿，float64经纬高+RGB点云，字段/顺序/往返验证；法线留在米制点云。 |
| GT隔离及评价实现 | ✅ 已实现 | GT只用于评价；六图召回固定分母、严格阈值、ddof=0；不以GT初始化、配准或调权。 |
| 全像素D/D0与分区 | ✅ 已实现 | 完整反投影，原版先冻结D0；同位置绝对高程误差、总体/联合/共同/新增/丢失区及NoData/覆盖诊断。 |
| 自动调度、断点与冻结保护 | ✅ 已实现 | all入口补齐未冻结缺失阶段，确认停点，GT冻结后禁止重建并复用核验；本轮逐阶段实跑及all复用均通过。 |
| 路径、计数适配和文档 | ✅ 已实现 | 本轮独立输出路径；未发现旧批数据路径/数量判据；使用说明第9节列明设置与协议边界。 |
| 计划9.3八项精度门槛 | ⚠️ 部分实现 | 评价功能已实现并完成实测，但仅4/8指标达标，旋转和DSM的median/std未达标；最终交付目标未完成。 |

### 输入和自动化边界

- 本期约定输入：六张RGB照片、DOM/DSM、固定K、真实原版SfM/MVS及前端特征/匹配和配套缓存。它们齐备且本批先验获明确确认后，可以自动运行到输出及GT评价。
- ❌ 未实现：仅提供裸图和地图就由当前入口自动生成原版SfM/MVS及全部原版前端文件；这不在本计划的既定输入范围内。
- ❌ 未实现：无GT时一条 `all` 命令运行至导出并自动跳过评价；当前 `all` 必须带评价配置，无GT可按阶段运行到export。
- 候选确认是计划规定的停点；“自动化”不表示可无授权选权重。换数据不能套用旧确认。

## 第二步：完整真实使用流程实测

本轮重新创建空的 `joint_map` 输出目录，没有复制中间产物；验收脚本逐项调用项目原有CLI，完整保存命令、stdout、退出码及耗时。下面是本轮实际执行的命令和输出节选，不是仅列复现建议。

### 2.0 单元与反例回归

```bash
TMPDIR=/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/tests_tmp /opt/anaconda3/envs/pixsfm/bin/python -B -m unittest discover -s tests -p 'test_joint_map.py' -v
```

```text
Ran 137 tests in 13.893s
OK
```

完整日志：[unit_tests.log](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/unit_tests.log)。动态数据回归包含 `test_dynamic_partition_sizes_and_reproducibility`、`test_workspace_accepts_input_order_and_dynamic_point_counts` 和 `test_configured_cache_creation_reuse_and_mismatch_refusal`，覆盖不同数量/网格、同名异内容缓存及非法输入拒绝。

### 运行配置和自动记录入口

本轮配置：[config.json](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/config.json)，与原配置仅两个输出位置字段不同；[setup.json](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/setup.json)记录输出开始为空。调用已有CLI的验收记录脚本为[acceptance_flow.py](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/acceptance_flow.py)，不会下载依赖或改变算法。

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/acceptance_flow.py
```

总记录：[flow_state.json](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/flow_state.json)、[flow.stdout.log](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/flow.stdout.log)。当前目录已GT冻结，再运行脚本会保留已完成记录；下面逐阶段重建命令只代表本轮首次执行，不应对已冻结目录强行重跑。

### 2.1 check

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_joint_map.py --config /media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/config.json --stage check
```

```text
check: PASS | images=6 | sparse_points=4280 | valid_dsm_pixels=154730
exit_code=0
```

耗时 1.732 秒；完整stdout：[日志](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/check.stdout.log)。

### 2.2 baseline

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_joint_map.py --config /media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/config.json --stage baseline
```

```text
baseline: complete | images=6 | map_tracks=0
exit_code=0
```

耗时 1.798 秒；完整stdout：[日志](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/baseline.stdout.log)。

### 2.3 initialize

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_joint_map.py --config /media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/config.json --stage initialize
```

```text
initialize: complete | images=6 | map_tracks=954
exit_code=0
```

耗时 15.632 秒；完整stdout：[日志](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/initialize.stdout.log)。

### 2.4 validate_priors

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_joint_map.py --config /media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/config.json --stage validate_priors
```

```text
validate_priors: complete | images=6 | map_tracks=943
exit_code=0
```

耗时 648.783 秒；完整stdout：[日志](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/validate_priors.stdout.log)。

本轮动态划分：训练18106条、留出784条，冻结943个训练地图先验。自动缓存 `reused=False`、`complete=True`。

| 候选 | 终态 |
|---|---|
| weight_0p1 | passed |
| weight_1p0 | passed |
| weight_10p0 | passed |

#### 确认停点的本轮处理

实际调用 `joint_map.optimization.confirm_prior_candidate(output_dir, candidate_id, user_confirmation)`。先核对本批顺序输入、非输出配置、策略、工作尺度、全部12组先验数组及留出身份/像素逐值一致，且1.0候选重新验证通过，才沿用用户此前对同批输入的明确1.0工程选择；没有使用GT挑权重。

确认记录：[confirmation_binding.json](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/confirmation_binding.json)。本轮 `candidate_id=weight_1p0`、`lambda_map=1.0`；换输入时不得复用此授权。

### 2.5 register

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_joint_map.py --config /media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/config.json --stage register
```

```text
register: complete | images=6 | map_tracks=943
exit_code=0
```

耗时 8.266 秒；完整stdout：[日志](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/register.stdout.log)。

### 2.6 joint

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_joint_map.py --config /media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/config.json --stage joint
```

```text
formal joint complete: images=6, points=4280, observations=18890, map_priors=943
joint: complete | images=6 | map_tracks=943
exit_code=0
```

耗时 144.830 秒；完整stdout：[日志](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/joint.stdout.log)。

### 2.7 dense/baseline

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_joint_map.py --config /media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/config.json --stage dense --branch baseline
```

```text
dense baseline: complete | images=6 | points=522615
dense: complete | images=6 | map_tracks=0
exit_code=0
```

耗时 13.670 秒；完整stdout：[日志](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/dense_baseline.stdout.log)。

### 2.8 dense/joint

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_joint_map.py --config /media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/config.json --stage dense --branch joint
```

```text
dense joint: complete | images=6 | points=508917
dense: complete | images=6 | map_tracks=0
exit_code=0
```

耗时 109.631 秒；完整stdout：[日志](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/dense_joint.stdout.log)。

### 2.9 update/baseline

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_joint_map.py --config /media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/config.json --stage update --branch baseline
```

```text
update baseline: complete | update_status=updated | cells=23409 | area_m2=2100.575709
update: complete | images=6 | map_tracks=0
exit_code=0
```

耗时 17.602 秒；完整stdout：[日志](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/update_baseline.stdout.log)。

### 2.10 update/joint

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_joint_map.py --config /media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/config.json --stage update --branch joint
```

```text
update joint: complete | update_status=updated | cells=23396 | area_m2=2099.409176
update: complete | images=6 | map_tracks=0
exit_code=0
```

耗时 17.698 秒；完整stdout：[日志](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/update_joint.stdout.log)。

### 2.11 export/baseline

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_joint_map.py --config /media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/config.json --stage export --branch baseline
```

```text
export baseline: complete | poses=6 | sparse_points=4280 | dense_points=522615
export: complete | images=6 | map_tracks=0
exit_code=0
```

耗时 8.547 秒；完整stdout：[日志](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/export_baseline.stdout.log)。

### 2.12 export/joint

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_joint_map.py --config /media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/config.json --stage export --branch joint
```

```text
export joint: complete | poses=6 | sparse_points=4280 | dense_points=508917
export: complete | images=6 | map_tracks=0
exit_code=0
```

耗时 8.560 秒；完整stdout：[日志](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/export_joint.stdout.log)。

### 2.13 evaluate

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_joint_map.py --config /media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/config.json --stage evaluate --evaluation-config /home/amax/Documents/code/pixel-perfect-sfm-main/configs/joint_map_evaluation.json
```

```text
evaluate: complete | D=49766333 | D0=49766286 | primary_acceptance=failed
evaluate: complete | images=6 | map_tracks=0
exit_code=0
```

耗时 114.849 秒；完整stdout：[日志](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/evaluate.stdout.log)。

### 2.14 all

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_joint_map.py --config /media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/config.json --stage all --evaluation-config /home/amax/Documents/code/pixel-perfect-sfm-main/configs/joint_map_evaluation.json
```

```text
all: complete | reused=13 | executed=0 | primary_acceptance=failed | delivery_complete=False
exit_code=0
```

耗时 9.987 秒；完整stdout：[日志](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/all.stdout.log)。

### 本轮实际产物和精度

| 项目 | 原版 | 联合 |
|---|---:|---:|
| 注册图像 | 6 | 6 |
| 稀疏点 | 4280 | 4280 |
| 图像观测 | 18890 | 18890 |
| 稠密点 | 522615 | 508917 |
| 更新像元 | 23409 | 23396 |
| 更新面积m² | 2100.5757086628582 | 2099.4091756972484 |

固定参考域D=49766333，主验收D0=49766286；新增无效参考点=0。std使用ddof=0；指标使用未舍入值判定。

| 八项主指标 | 基线 | 联合 | 是否达标 |
|---|---:|---:|---|
| translation_median_m | 0.22175004683298347 | 0.2215410149687538 | ✅ |
| translation_std_m | 0.007694962250426519 | 0.007374102315788271 | ✅ |
| rotation_median_deg | 0.1288528516901784 | 0.13464276373985373 | ❌ |
| rotation_std_deg | 0.0007974567234481662 | 0.0009787643833574632 | ❌ |
| dsm_median_m | 0.03881753835700508 | 0.039765176803653546 | ❌ |
| dsm_std_m | 5.0704745214072275 | 5.070775575165666 | ❌ |
| recall_1m_1deg | 1.0 | 1.0 | ✅ |
| recall_3m_3deg | 1.0 | 1.0 | ✅ |

| 诊断分区 | 参考点数 | 配对median：基线/联合m | 配对std：基线/联合m | 存在退化 |
|---|---:|---|---|---|
| all | 49766333 | 0.03881753835700508 / 0.039765176803653546 | 5.0704745214072275 / 5.070775575165666 | True |
| union | 36889274 | 0.027756760574929373 / 0.028437680188389436 | 5.164872770581885 / 5.165332490174848 | True |
| common | 34857567 | 0.026207561262538093 / 0.02689007218771522 | 5.194167073198784 / 5.194602422706397 | True |
| new | 985924 | 0.4011605831080072 / 0.3650525177691133 | 4.637850110492416 / 4.584969952564913 | False |
| lost | 1045783 | 0.381234774055514 / 0.4334964042422982 | 4.611405422587937 / 4.6623211412697145 | True |

完整评价：[report.md](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/joint_map/evaluation/report.md)、[report.json](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/joint_map/evaluation/report.json)。原图NoData新增区、无配对和D外覆盖详见JSON；分区是诊断，不增加八项之外的硬门槛。

## 第三步：问题排查和等级

**本轮功能性P0：0。** 14条CLI均退出0，13个计算阶段及全部交接完成。另有4个精度硬门槛未达到；它们是计划第9.3节的验收阻断项，不能降级为P1体验问题。

| 编号 | 等级 | 发现的问题 | 本次处理及剩余情况 |
|---|---|---|
| ACC-01 | 精度硬门槛阻断 | 旋转median/std及DSM绝对误差median/std高于基线 | 如实判未通过；按计划10.3/第9节后的约束保持冻结结果，改进设计及独立实验需用户决定 |
| UX-01 | P1 | output_dir与selected_model_path须同步，误设会停止 | 已补同时设置两字段的示例；严格归属检查保留 |
| UX-02 | P1 | 原版配套缓存/summary等并非全部在check阶段提前检查，缺失可能较晚暴露 | 已补完整目录清单和前置条件；集中提前总校验仍是体验改进项 |
| UX-03 | P1 | all在候选未确认时抛ValueError并带traceback，提示不够友好 | 已说明这是确认停点，补确认API及续跑步骤；当前异常协议保留 |
| UX-04 | P1 | 通用CLI对未上报先验数量的阶段可能打印map_tracks=0，容易误读 | 已指明以冻结NPZ、map_prior_count和集成accepted_map_priors为准；通用显示统一仍可改进 |

固定六图/K/地图格式和原版输入范围属于本期协议边界，不按“功能实现了但任意输入仍可用”来宣传。

## 第四步：P0修复与复测

- 未发现本期约定输入下功能跑不通的P0，因此没有为验收修改生产算法代码或重装库；本次修改为使用说明、验收文档及验收记录脚本。
- 本轮重新运行137项回归全部通过，并实际从空输出运行13阶段；最后all复用13阶段并完成完整核验。所有命令、输出和退出码见第二步。
- P1说明缺口已在使用文档第9节补齐；交互式确认、统一前置检查及计数显示的代码体验改进未混入本次功能修复。
- 4项精度失败仍存在；根据计划不进行看到GT后的自动调权、换关联或放宽阈值。工程流程通过与整体精度验收未通过分开记录。

## 第五节：写死数据专项审查

检查范围：10份应用入口/模块Python源码，结果见[hardcoding_audit.json](/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/hardcoding_audit.json)。不将配置中的本批路径、文档实测数字或合成自测场景常数混为旧数据运行判据。

| 检查项 | 结果与依据 |
|---|---|
| 旧批目录回退 | 未发现生产源码直接使用旧批/media或/home数据路径，旧original_pixsfm目录不作为隐藏fallback；路径来自配置 |
| 旧/当前批观测数与先验数判据 | 未发现777/18013/942等历史数字或本批784/18106/943等结果数用作运行常量；N按当前图像tracks、划分和筛选生成 |
| 写死的协议：六图和K | **存在且保留**：run_joint_map.py的check_config固定六图、3840×2160及K=[2700,2700,1920,1080]，符合本期计划 |
| 写死的特征配置 | **存在且保留**：optimization.get_dense_provenance固定900×1600×128 float16、S2DNet；由本期3840×2160和1600最大边长契约限定 |
| 固定算法规则 | **存在且保留**：候选0.1/1/10、留出0.2/seed0、300/100预算、原容差、2 m共同筛选及可靠性/分层规则；不是旧数据数量 |
| 固定解释器/系统资源路径 | **存在且保留**：/opt/anaconda3/envs/pixsfm按计划限定；/proc/meminfo用于资源检查，不是数据目录 |
| 固定配套文件名与后端 | run_config.json、dense_features.h5、COLMAP/NPZ/PLY协议，以及reuse_registered_sfm/SuperGlue唯一实现；从本批配置目录派生 |
| 配置中的绝对目录 | configs/joint_map.json明确保存本批实际目录；可换成本批真实输入，不需改算法；不能只换照片而仍引用旧模型/特征 |

**结论：没有发现旧批数据数量或旧目录回退依赖，但仍有明确固定协议，不能称“完全没有硬编码”或“任意输入只改JSON就能跑”。**

## 第六节：使用说明与常规数据设置

完整说明：[USAGE_JOINT_MAP.md](/home/amax/Documents/code/pixel-perfect-sfm-main/USAGE_JOINT_MAP.md)第9节含逐键设置、目录树、配置生成示例及确认方法。

| 设置位置 | 配置键/文件 | 规则 |
|---|---|---|
| 重建配置 | configs/joint_map.json，或另存为configs/my_batch.json | 只放重建输入，不得含GT/evaluation字段 |
| 照片 | image_dir、image_names | 六张RGB、3840×2160；名字/顺序匹配原版与GT；名称不含空白，GT按不同stem配对 |
| 地图 | dom_path、dsm_path | 共网格EPSG:4326；DOM三波段int16/NoData255，DSM float32/NoData-9999；高度为约定WGS84椭球高 |
| 原版输入 | native_sfm_dir、native_dense_dir、frontend三路径 | 本批真实SfM/MVS和HDF5；附run_config.json、dense_features.h5及完整稠密工作区，不能套旧批 |
| 新输出 | output_dir | 建独立目录；本轮输出路径见下表 |
| 本批确认文件 | prior_validation.selected_model_path | 必须同步为output_dir/shared/prior_model.json |
| GT | configs/joint_map_evaluation.json中的gt_pose_path、gt_depth_dir | 六行无表头pose，与image_names同序；同名stem.npy、float32、2160×3840相机z深度米制 |
| 相对路径 | 所有重建/评价配置路径 | 相对于项目根目录，而不是JSON所在目录；绝对路径可直接使用 |
| camera/候选/分辨率 | camera、prior_validation | 当前固定协议；其他K、图像数、深度语义或候选不能随意只改JSON |

| 本次目录 | 实际位置 |
|---|---|
| 重建验收配置 | `/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/config.json` |
| 验收输出 | `/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/joint_map` |
| 原始照片 | `/home/amax/Documents/code/pixel-perfect-sfm-main/datasets/images/DJI_20251221132525_0004_V` |
| DOM / DSM | `/home/amax/Documents/code/pixel-perfect-sfm-main/datasets/maps/dom.tif` / `/home/amax/Documents/code/pixel-perfect-sfm-main/datasets/maps/dsm.tif` |
| 原版SfM / MVS输入 | `/media/amax/PS2000/pixelsfm/outputs/new_data_20261110/original_pixsfm/sfm` / `/media/amax/PS2000/pixelsfm/outputs/new_data_20261110/original_pixsfm/dense` |
| 原实验保留目录 | `/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/joint_map` |

所有新运行缓存、日志和实验结果均在指定 `/media/amax/PS2000/pixelsfm/outputs` 下；源代码和文档仍在项目目录。

### 最终产品在哪里

- `<output_dir>/baseline|joint/export/poses_wgs84.txt`：六行经纬高/roll/pitch/yaw。
- `<output_dir>/baseline|joint/export/{sparse_wgs84,dense_wgs84}.ply`：double经纬高与RGB。
- `<output_dir>/baseline|joint/map/updated_dsm.tif`、四种mask.tif：原网格DSM及证据掩膜。
- `<output_dir>/joint/map_point_lineage.npz`及baseline/map/sparse_evidence.npz：身份和更新链。
- `<output_dir>/evaluation/report.md|report.json`：八项主验收、全部分区和无值统计。
- `<output_dir>/integration.json`、run_manifest.json：交接、执行、精度和交付状态，不能只看退出码。

### 当前结果如何手动复核

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B run_joint_map.py --config /media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/config.json --stage all --evaluation-config configs/joint_map_evaluation.json
```

当前已冻结完整结果应复用13阶段，不重跑重建或重新选权。查看integration的implementation_complete=true、primary_acceptance_passed=false及delivery_complete=false；再查evaluation/report.json中的8项逐条比较。

## 最终判定

**功能实现及约定输入下的完整流程：通过。计划要求的八项精度及整体MVP/最终交付：未通过。** 原始数据、阈值和失败结果保留，未实施加分项。是否调整方案开展下一轮独立精度实验，由用户决定。

验收附件统一位于 `/media/amax/PS2000/pixelsfm/outputs/redevelopment_20261113/p4_acceptance_20261114/`：

- `acceptance_audit.json`：最终功能/精度判定、摘要校验、逐点误差汇总复核及文档SHA。
- `artifact_inventory.json`：本轮输出文件、大小及SHA-256完整清单。
- `final_recheck.stdout.log`：文档定稿后的冻结结果复用核验。
- `flow_state.json`及各阶段 `*.stdout.log`：14条实际CLI命令和原始stdout。

历史模块9验收记录是当时文件版本的快照，不回写其中的旧文档摘要；本轮使用更新后的使用说明及以上独立P4记录。
