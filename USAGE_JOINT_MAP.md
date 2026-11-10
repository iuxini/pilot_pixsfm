<!-- 用途：记录固定六图流水线的运行方式、交接协议、实测结果及尚未通过的精度验收。 -->
# 六图地图引导 PixSfM 使用与交付记录

本批工程模块已经实现；真实 GT 八项主验收有四项未通过，整体精度目标未完成。两分支均产生了有效 DSM 更新。`status=complete` 表示阶段执行完成，不能解释为精度通过。

## 1. 运行环境与输入

所有 Python 命令仅使用 `/opt/anaconda3/envs/pixsfm/bin/python`。本轮不安装、升级库或修改环境设置。

| 项目 | 已验证值或路径 |
|---|---|
| 项目目录 | `/home/amax/Documents/code/pixel-perfect-sfm-main` |
| 重建配置 | [configs/joint_map.json](configs/joint_map.json) |
| 独立评价配置 | [configs/joint_map_evaluation.json](configs/joint_map_evaluation.json) |
| 六图 | `datasets/images/DJI_20251221132525_0004_V/{0..5}_0.png` |
| GT 深度，仅评价读取 | 同目录六份同名 `.npy` |
| GT 位姿，仅评价读取 | `datasets/poses/DJI_20251221132525_0004_V.txt` |
| DOM/DSM | `datasets/maps/dom.tif`、`datasets/maps/dsm.tif` |
| 原版稀疏/稠密产物 | `outputs/original_pixsfm_20261108/sfm`、`dense` |
| CUDA COLMAP | `build/colmap_cuda/install_ptx90/bin/colmap`，3.10-dev，PTX 90 构建 |
| Python 库 | NumPy 1.26.4、SciPy 1.13.1、Rasterio 1.4.3、PyProj 3.6.1、h5py 3.14.0 |
| 前端/模型读写 | OpenCV 4.10.0、Torch 2.8.0+cu128、PyCOLMAP 0.4.0、hloc 1.4、Pillow 11.3.0 |

PixSfM 与 pyceres 使用现有安装。先前已获授权补齐的 pyceres 接口为真实 `set_num_residuals`、安全求值桥接、原生 QuaternionManifold `plus` 和 `plus_jacobian`。源码位于 `build/pixsfm_env_sources/pyceres/_pyceres/core/`；原扩展备份为 `build/pyceres_module3/pyceres.original.so`。模块9没有再次构建扩展或修改依赖。

PyCOLMAP 负责模型读写；MVS 使用上表独立 CUDA COLMAP。需要实际 GPU 重建时，应在原来可访问 RTX 5090 的执行会话运行。沙箱内驱动不可见不等于主机驱动损坏；当前集成检查复用已完成的 CUDA 产物。

## 2. 坐标、相机和文件约定

图像分辨率为 3840×2160，固定 PINHOLE 参数 `[2700,2700,1920,1080]`；优化中内参固定。内部几何使用 `shared/origin.json` 定义的 ENU 米制坐标，原点来自输入地图，不来自 GT。

COLMAP 使用 `X_camera=R_cw*X_world+t_cw`，相机中心为 `-R_cw.T*t_cw`，四元数顺序为 `qw qx qy qz`。共同地图变换为 `X_ENU=scale*rotation*X_SfM+translation`；基线深度乘相同 scale，法线只旋转。

位姿文本每行是 `image_name longitude latitude ellipsoidal_height roll pitch yaw`，共六行，角度为度，`pitch=0` 表示俯视。姿态采用每个相机地理位置处的 ENU 基底：`R_render_to_ENU=Rotation.from_euler('xyz',[pitch,roll,yaw],degrees=True)`，CV/渲染轴转换为 `diag(1,-1,-1)`。

地理 PLY 的 `x/y/z` 是经度、纬度、WGS84 椭球高；XYZ 为 double，带 RGB，头部写明 EPSG:4326 与单位。法线保留在米制 PLY；经纬度 PLY不作为欧氏距离、优化或法线计算的输入。稀疏点按 point3D_id 升序导出；稠密点保留原融合点顺序，不按 GT 或误差过滤导出点。

输入 DSM 为 EPSG:4326、395×404、float32，NoData=-9999。像元分桶用包含位置的原网格像元；采样使用像元中心的四邻域双线性插值，任一所需邻居无效则记录无值。未更新位置的原值与 NoData 保持不变。

## 3. 数据流与地图作用

```text
六图 + 固定 K -> 原版特征/匹配/相对 SfM -> 共享 tracks
DOM 变体匹配 + DSM -> 身份回溯/几何筛选 -> 训练地图先验 M_train
                                         |
条件留出验证 -> 有效候选 -> 用户确认 -> 共同地图 Sim(3)
                                         |
                  +----------------------+---------------------+
                  |                                            |
       原版相对几何只做共同变换                   地图点 XYZ=mu 初始化
       原版融合点云/深度同步变换                   几何 -> 特征联合优化
                  |                                            |
                  |                                  优化相机 + 稀疏点 -> CUDA MVS
                  +----------------------+---------------------+
                                         |
                     两分支点级可靠性 -> 分层/冲突检查 -> DSM 更新
                                         |
                               WGS84 位姿与点云导出
                                         |
GT pose/depth ---------------------> 独立评价：先冻结基线/D0，再计算联合
```

DOM 单应性只作二维粗筛：变体坐标还原至原 DOM 后，前向重投影阈值为3 px；地图关联最终还要经过 DSM 三维检验，共同 Sim(3) 的三维阈值为2 m。允许单图 DOM 候选，但须具有可靠多视图 track、正确描述子/坐标/身份，并由不包含该候选的可靠关联建立独立 Sim(3) 验证；证据不足保留待验证。

模块3冻结训练观测18,013条、留出777条和942个地图先验；地图投票与训练参考排除留出。原版初值使用完整观测，因此 `initialization_uses_heldout=true`，留出结果仅解释为条件稳定性，不称为独立泛化精度。

视觉参考模型用于估计参考描述子和噪声尺度；优化初始模型的地图关联 XYZ 逐点赋为 mu，两者用途不同。每条 track 始终只有一个共享 XYZ；图像与地图项使用同一可变参数块，六个位姿全部开放，先验持续存在。地图残差小、点移动或总目标下降不能证明绝对精度提高。

`shared/holdout_dense_features.h5` 沿用原文件名，含六张最大边长1600、128通道、float16整幅特征；经同指纹、数值和边界适配核验后供训练/留出采样。原版补丁缓存保持不变。正式阶段恢复全部18,790条图像观测，重新建立全观测参考，但保留同一942个训练先验及其均值、协方差、损失和确认权重。

## 4. 运行入口与冻结条件

在项目目录执行完整检查或继续未完成阶段：

```bash
cd /home/amax/Documents/code/pixel-perfect-sfm-main
/opt/anaconda3/envs/pixsfm/bin/python -B -u run_joint_map.py --config configs/joint_map.json --stage all --evaluation-config configs/joint_map_evaluation.json
```

`all` 按 check、baseline、initialize、validate_priors、register、joint、两分支 dense、两分支 update、两分支 export、evaluate 共13个阶段/分支记录执行。已完成阶段复用，最后核验交接、实际数据、SHA-256与验收状态。评价配置仅在最后的评价阶段使用，不传入重建阶段。

当前13个阶段/分支都已执行完成，所以这条命令应复用全部结果，只做集成核验，不重新优化、MVS或计算GT指标。结果写入 `outputs/joint_map/integration.json` 与 `logs/all.log`。

`validated` 不等于 `confirmed`。候选未获用户明确确认时，流水线停止在 register 之前，不自动选权重。本次用户选择1.0，记录为 `user_selected`，不宣称数据证明1.0最优。

GT 基线已经冻结时，当前输出目录禁止重建阶段重跑，包括 check、重新关联、验证先验、优化和改写掩膜。`all` 只复用完整阶段；若某个前置阶段缺失或最新尝试失败，会停止。新的实验需先明确授权并使用独立输出和评价记录，不能覆盖当前失败结果后重新选择参数。

以下是尚未冻结的新实验所用的阶段接口清单，不能对当前已冻结目录从头重跑：

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B run_joint_map.py --config configs/joint_map.json --stage check
/opt/anaconda3/envs/pixsfm/bin/python -B run_joint_map.py --config configs/joint_map.json --stage baseline
/opt/anaconda3/envs/pixsfm/bin/python -B run_joint_map.py --config configs/joint_map.json --stage initialize
/opt/anaconda3/envs/pixsfm/bin/python -B run_joint_map.py --config configs/joint_map.json --stage validate_priors
# 此处必须已获得用户的候选确认；命令不会自动确认。
/opt/anaconda3/envs/pixsfm/bin/python -B run_joint_map.py --config configs/joint_map.json --stage register
/opt/anaconda3/envs/pixsfm/bin/python -B run_joint_map.py --config configs/joint_map.json --stage joint
/opt/anaconda3/envs/pixsfm/bin/python -B run_joint_map.py --config configs/joint_map.json --stage dense --branch baseline
/opt/anaconda3/envs/pixsfm/bin/python -B run_joint_map.py --config configs/joint_map.json --stage dense --branch joint
/opt/anaconda3/envs/pixsfm/bin/python -B run_joint_map.py --config configs/joint_map.json --stage update --branch baseline
/opt/anaconda3/envs/pixsfm/bin/python -B run_joint_map.py --config configs/joint_map.json --stage update --branch joint
/opt/anaconda3/envs/pixsfm/bin/python -B run_joint_map.py --config configs/joint_map.json --stage export --branch baseline
/opt/anaconda3/envs/pixsfm/bin/python -B run_joint_map.py --config configs/joint_map.json --stage export --branch joint
/opt/anaconda3/envs/pixsfm/bin/python -B run_joint_map.py --config configs/joint_map.json --stage evaluate --evaluation-config configs/joint_map_evaluation.json
```

## 5. 求解、边界与可靠性规则

几何阶段最多300次，联合特征阶段最多100次；沿用 function_tolerance=1e-6、gradient_tolerance=1e-10、parameter_tolerance=1e-8，要求原容差收敛。保存每次代价/变化、梯度范数、步长、信赖域半径和具体终止消息；初始评估记为第0条，不算优化更新。

几何阶段只在除法前检查有限性和正深度，允许图外投影。联合特征阶段每次求值前还检查严格插值域；无效试探向 Ceres 返回失败，不采样、不夹坐标、不补零、不删因子。几何收敛后若特征域无效，按预定 beta=[1,10,100]、每级最多100次恢复可行性；最终联合问题必须移除恢复项。

方向导数核验策略为 `multiscale_richardson_roundguard_v1`，使用原生 manifold 的 Plus 与 plus_jacobian、固定多尺度序列、128通道公共窗口和舍入保护，区分通过、不一致与未决。正式因子只返回环境空间 Jacobian，交给 Ceres 应用 manifold。

MVS沿用原版参数：PatchMatch最大边长2000、几何一致性与一致性图启用、GPU 0、cache_size=2；融合使用 geometric 输入、最大边长2000、8线程。`.vis` 的 image_idx 按原生 `images.bin` 读取顺序恢复真实图像身份，不能用文件名或 image_id 排序代替。

点级更新准入：实际正深度和图内可见、至少3个不同支持图像、至少一对视差角≥1°、双向深度反投影重投影误差≤2 px、相对深度误差≤1%、法线夹角≤10°。K按实际深度分辨率缩放。遮挡先判定；深度一致性带为预测 z 的1%，不是估计协方差。

对通过点级检查的点，按原 DSM 像元分组：

1. 椭球高排序，相邻差≥1.0 m分组。每组至少3个不同稠密点才有支持。
2. 至少两个有支持层：`multiple_height_layers`，优先级最高。
3. 否则若仍存在多个分离组：`ambiguous_height_support`，不丢弃小组后更新。
4. 单组继续要求至少3点且未缩放 MAD≤0.5 m，更新为该组椭球高中位数。

1.0 m为本批预先固定工程阈值，不能识别所有高度重叠或连续连接的多层表面。至少两个有效、不同支持视图一致显示稀疏—稠密冲突时，只禁止共享点优化后所在的一个像元；多个冲突取并集，不膨胀邻域。网格外位置记录为未映射，不裁剪坐标。

四掩膜满足 `update_mask=coverage_mask & overlap_mask & reliability_mask`。无可靠稠密支持的稀疏移动不强写 DSM。身份链为 `track -> 共享XYZ初值/结果 -> 优化模型 -> 同像元可靠稠密ID/支持视图 -> 实际更新像元`。

## 6. 评价规则与本批结果

本批六组深度按用户明确确认，仅使用 `np.isfinite(depth) & (depth>0)`，不按far额外过滤，不删除最大值。确认、照片/深度配对、尺寸和文件指纹写入 `evaluation/reference_manifest.json`；此规则只适用于其中标识的六组文件。

D由GT有效性及输入地图范围定义；按图像顺序、row_major保留所有像素身份，不抽样、不去重。D0取D中基线最终DSM采样有效的参考点，包含保留原图的位置。先冻结 `baseline_metrics.json`、D0身份和输入摘要，再计算联合分支，禁止用联合掩膜缩减D0。

DSM比较同一位置的双线性采样高程与GT反投影椭球高，误差为绝对高度差。std均为ddof=0；两个Recall分别用严格 `<1 m && <1°`、`<3 m && <3°`，分母固定为6。八项使用未舍入值判定，六项误差不能高于基线，两个召回不能低于基线。

| 主指标 | 基线 | 联合 | 通过 |
|---|---:|---:|---|
| 平移 median m | 0.3007280134431326 | 0.3001987968731587 | 是 |
| 平移 std m | 0.01843289442248755 | 0.018218429851259647 | 是 |
| 旋转 median ° | 0.08878216938937875 | 0.09126165904634259 | 否 |
| 旋转 std ° | 0.0007782709340146194 | 0.0012869571216202293 | 否 |
| DSM median m | 0.022376727805980323 | 0.022785455033002222 | 否 |
| DSM std m | 4.299014110464808 | 4.300296413042522 | 否 |
| Recall@1m1deg | 1.0 | 1.0 | 是 |
| Recall@3m3deg | 1.0 | 1.0 | 是 |

D与D0均为49,766,400个参考点，无联合有效性退化。分区另报总体、联合、共同、新增、丢失区的配对误差、原图误差、有效/无效数、像元和米制面积。新增区median由0.18839898348722173 m增至0.20430037064191353 m，std由4.120536793595799 m降至4.0585768781118325 m。这是独立诊断，不新增主验收门槛。

| 产物计数 | 基线 | 联合 |
|---|---:|---:|
| 位姿 / 稀疏点 | 6 / 4254 | 6 / 4254 |
| 融合稠密点 | 523872 | 538534 |
| 几何深度 | 六张2000×1125 | 六张2000×1125 |
| 更新像元 | 23371 | 23295 |
| 更新面积 m² | 2097.165833714651 | 2090.3460785869975 |
| 稀疏冲突像元 | 6 | 8 |

原图NoData新增填补区如无基线配对，只报联合绝对误差及有效覆盖，不新增精度门槛；本批该区参考点数为0。D以外的更新面积仅报告覆盖与重建质量，不宣称完成GT验收。GT本身是渲染参考，不宣称独立实测控制点精度。

## 7. 输出、状态与失败记录

所有路径位于 `outputs/joint_map/`。原始图像、地图、原版缓存和模型不被覆盖。

| 路径 | 用途 |
|---|---|
| `run_manifest.json` | 各次阶段状态、计数、产物；保留历史失败尝试 |
| `integration.json`、`logs/all.log` | 完整交接核验、复用/执行阶段、文件摘要及交付状态 |
| `shared/prior_model.json` | confirmed状态、用户确认、权重及冻结先验摘要 |
| `shared/prior_validation.json` | 数值桥接、候选、777条留出和条件验证证据 |
| `shared/candidates/<run_id>/` | 模块3冻结输入及三候选独立结果 |
| `joint/runs/<run_id>/` | 模块4初始化、实际求解结果、全观测参考和诊断快照 |
| `<branch>/model/` | ENU米制COLMAP模型，6图、4254点、18790条观测 |
| `<branch>/dense/` | `fused_metric.ply`、`.vis`、`image_order.json`及summary |
| `<branch>/map/` | `updated_dsm.tif`、四掩膜、`point_evidence.npz`和summary |
| `joint/map_point_lineage.npz` | 942个共享点的原图位置、初值、结果、深度/视图证据、稠密ID及输出像元 |
| `baseline/map/sparse_evidence.npz` | 原版共享点的独立稀疏—稠密证据 |
| `<branch>/export/` | 六行 `poses_wgs84.txt`、double稀疏/稠密PLY及summary |
| `evaluation/baseline_metrics.json` | 不覆盖的基线主指标、D0身份与来源 |
| `evaluation/reference_points.bin` | 按manifest dtype解释的全量参考身份/坐标，不能按普通PLY读取 |
| `evaluation/dsm_errors_*.npy` | D上的float64逐点绝对误差，无值为NaN且另存有效性掩码 |
| `evaluation/dsm_partition_masks.npz` | 五分区、D0及各方法有效性/NoData标记 |
| `evaluation/pose_errors.csv` | 六图两分支逐图平移/旋转误差 |
| [evaluation/report.md](outputs/joint_map/evaluation/report.md)、[report.json](outputs/joint_map/evaluation/report.json) | 人可读报告及全精度机器结果 |

`status=complete` 只表示运行结束；`implementation_complete` 表示集成检查通过；`accuracy_acceptance_passed` / `primary_acceptance_passed` 表示八项主验收；`delivery_complete` 还要求两分支实际产生更新。本批后两类精度/交付标志均为false。

`update_status=updated` 表示有可靠点支持并写入更新；`no_valid_update` 表示掩膜为空、零数量和零面积，保留原图。空更新可以继续导出/评价，但更新目标必须保持未交付，不能靠文件生成或GT主指标通过掩盖它。

完整stdout/stderr保存在各阶段日志：`logs/check.log`、`baseline.log`、`initialize.log`、`validate_priors.log`、`register.log`、`joint.log`、`dense_baseline.log`、`dense_joint.log`、`update_baseline.log`、`update_joint.log`、`export_baseline.log`、`export_joint.log`、`evaluate.log`。联合COLMAP日志位于 `joint/dense/{undistort,patch_match,fusion}.log`。

已修复的观测核验误报来自原生去畸变像素往返舍入；检查现按相同原生变换精确比较。模型四元数保存/读取各归一化一次，按两次原生归一化后的预期值精确核验，没有放宽优化容差。旧失败诊断和 `baseline/dense/attempt1_failure.log` 保留。

原生求解失败或输入摘要不符时先查看对应阶段日志与最后checkpoint。没有结果的阶段不能仅凭文件存在认定完成。同一bug最多尝试修复3次；达到上限或出现方案不确定项就停止说明。四项GT主指标未通过属于当前实验结论，不在本轮自动搜索新的地图权重、关联或掩膜参数。

## 8. 测试与核验

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -m unittest discover -s tests -p 'test_joint_map.py' -v
/opt/anaconda3/envs/pixsfm/bin/python -B -u run_joint_map.py --config configs/joint_map.json --stage all --evaluation-config configs/joint_map_evaluation.json
```

测试包括坐标/姿态/双线性采样、身份/多视图连接、真实Ceres桥接、公共多尺度导数窗口、共享可动参数、边界拒绝/恢复、候选循环与确认、稠密索引、DSM反例和单像元冲突、空更新、double地理输出、固定D0与严格召回、主指标和新增区退化分离，以及完整入口的冻结/复用/GT隔离。

模块8工程回归实测97项通过；模块9新增6项集成测试，完整103项回归实测4.229秒通过。模块9实际集成核验以 `integration.json` 为准；该核验不重新计算GT分数。

已验证范围是本批固定六图、输入地图、现有权重和工具链。未验证其他地图噪声、其他场景或额外实验的稳定性；本批绝对精度四项仍未通过。内存/磁盘不足应停止，不自动降低分辨率、精度或可靠性阈值。

阶段API：`run_all(config_path,evaluation_config_path)` 顺序补齐未冻结的缺失阶段并复用完成结果；`audit_pipeline(config_path,evaluation_config_path,output_dir)` 只读核验完整交接与数据，分别返回工程、精度和交付状态。
