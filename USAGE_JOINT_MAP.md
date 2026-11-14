<!-- 用途：说明原版、改进版、误差评价三个独立入口及同批自动交接。 -->
# 三个独立入口：原版、改进版、评价

现在按顺序执行三个命令。第一步提供六张照片、DOM/DSM、输出根目录和内参；后两步通过同一个输出根目录自动读取本批交接，不需要手写配置JSON或填写中间文件路径。GT只传给第三步。

每组数据选一个输出根目录，下面统一用 `/media/amax/PS2000/pixelsfm/outputs/my_batch_1114` 举例。三个命令中的该路径保持一致。

三个入口均新增 `--mode real|modify`，默认 `real`。下面三个命令不加此参数就是正式运行：第一步从本次输入重新生成，后两步只共享该批新生成的结果，评价也重新建立GT参考，不查询开发缓存。固定数据调试时，在三个命令末尾都加 `--mode modify`；首次生成并建立缓存，后续按依赖校验复用。

## 1．只运行原版 PixSfM

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_original.py \
  --image-dir /home/amax/Documents/code/pixel-perfect-sfm-main/datasets/images/DJI_20251221132525_0004_V \
  --dom /home/amax/Documents/code/pixel-perfect-sfm-main/datasets/maps/dom.tif \
  --dsm /home/amax/Documents/code/pixel-perfect-sfm-main/datasets/maps/dsm.tif \
  --output-dir /media/amax/PS2000/pixelsfm/outputs/my_batch_1114 \
  --intrinsics 2700 2700 1920 1080
```

换新数据时，只替换照片目录、两张地图、输出根目录和四个内参。`--image-dir`应直接包含六张照片，不能填只有子目录的上一级；内参依次为`fx fy cx cy`。

这一步重新生成原版SuperPoint/SuperGlue、PixSfM关键点调整/SfM/特征BA、原版CUDA稠密工作区，然后完成用于公平比较的外部地图匹配、共同一致性筛选、先验尺度/候选标定、共同配准及原版后处理和导出。

**这里的地理原版结果是“原生PixSfM相对几何＋外部DOM/DSM配准”。原生PixSfM自身输出任意尺度的局部SfM坐标，不能称为原生WGS84定位。** 外部配准保留原版相对几何；候选标定用于冻结两版共用的地图模型，不将标定候选当作原版输出。这一步不执行正式改进版联合优化、不生成改进版稠密输出，也不读取GT或计算误差。

第一步完成后会停止，不会自动执行第二步或第三步。

## 2．只运行改进版 PixSfM

第一步成功后，单独执行：

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_improved.py \
  --output-dir /media/amax/PS2000/pixelsfm/outputs/my_batch_1114
```

这一步读取**当前批次**的原版初值、共同地图配准、冻结集合和缓存，重新生成改进版联合优化、CUDA稠密重建、DSM更新和WGS84导出。不会重新执行原版，不会修改原版结果，也不会读取GT或启动评价。

共用同批初值、地图先验和配准是本计划的公平对照要求；它不是从别的批次暗中找旧模型。缺少当前第一步结果、结果被改动或批次不一致时停止，不回退到更早的成功运行。候选仍按无GT策略自动选择，不要求用户调用确认API。

第二步完成后会停止，不会自动执行第三步。

## 3．只评价两版结果

前两步成功后，单独提供GT：

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/evaluate_results.py \
  --output-dir /media/amax/PS2000/pixelsfm/outputs/my_batch_1114 \
  --gt-poses /home/amax/Documents/code/pixel-perfect-sfm-main/datasets/poses/DJI_20251221132525_0004_V.txt \
  --gt-depth-dir /home/amax/Documents/code/pixel-perfect-sfm-main/datasets/images/DJI_20251221132525_0004_V
```

这一步只读取两版已完成结果、原图/原DSM和GT，生成新的评价目录。不会调用重建、配准、优化、稠密重建或DSM更新，也不会改写两版结果。没有GT时可先完成前两步，不能据此宣称绝对精度通过。

**已有旧版合并入口的完整结果时，无需重跑前两步。** `--output-dir`可以直接指定含`localization_result.json`的旧`run_...`目录，也可以指定仅含一个旧`run_...`结果的父目录；评价入口会检查两版导出与完成状态后读取它，不伪造或补写`batch.json`。例如旧结果保存在`my_batch_1114/run_...`时，上面的评价命令可以保持`--output-dir .../my_batch_1114`不变。若父目录有多个旧结果，必须指定准确的`run_...`目录，不会自动挑最新或跳过失败记录。存在新版`batch.json`时仍严格使用该批记录，失败或不完整时不回退到旧结果。旧结果的预测文件与输入指纹仍由原评价逻辑核验。

GT位姿为无表头的六行，顺序与第一步图像文件名排序一致：

```text
image_name longitude_deg latitude_deg ellipsoidal_height_m roll_deg pitch_deg yaw_deg
```

GT深度为六份同名stem加`.npy`的float32数组，形状2160×3840，含义为相机z方向深度，单位米。有效规则固定为`isfinite(depth) & (depth > 0)`，不额外排除far或最大值；GT数据须符合该协议。GT只参与评价，不用于重建、地图配准或权重选择。

报告包含平移误差median/std、旋转误差median/std、Recall@1m/1°和Recall@3m/3°、DSM同位置绝对高程误差median/std，以及逐图位姿误差和DSM总体/联合/共同/新增/丢失区诊断。std采用ddof=0，召回分母固定六图、阈值严格小于；DSM使用全部有效GT像素反投影，先冻结基线D0，再计算改进版。两版沿用同一地图配准，不用GT重新对齐。

## 输出与重新运行

程序自动管理当前批次的`batch.json`，不要手工修改。目录结构：

```text
<output-dir>/
├── batch.json
├── real/                             # 正式模式，每次新建run
│   ├── original/run_<UTC时间戳>/
│   │   ├── result.json
│   │   ├── reconstruction.json
│   │   ├── native/                   # 本批从照片生成的原版产物
│   │   ├── shared/                   # 两版共用的冻结标定、配准和缓存
│   │   └── baseline/
│   │       ├── export/poses_wgs84.txt # 原版地理位姿
│   │       ├── export/*_wgs84.ply
│   │       └── map/updated_dsm.tif
│   ├── improved/run_<UTC时间戳>/
│   │   ├── result.json
│   │   └── joint/
│   │       ├── export/poses_wgs84.txt # 改进版地理位姿
│   │       ├── export/*_wgs84.ply
│   │       └── map/updated_dsm.tif
│   └── evaluation/run_<UTC时间戳>/
│       ├── result.json
│       └── evaluation/
│           ├── report.md             # 人可读误差报告
│           ├── report.json           # 完整精度与验收状态
│           ├── pose_errors.csv
│           └── dsm_errors_*.npy
├── modify/                           # 开发模式，同样分original/improved/evaluation
└── modify_cache/
    ├── reconstruction/               # 原版、初始化、特征、共同准备和原版后处理清单
    └── evaluation/                   # GT参考与基线误差清单，重建入口不能使用
```

每个入口会打印自己的本次目录和结果路径，各自日志位于本次目录的`logs/step_*.log`或`logs/evaluate.log`。

`real`模式重新执行第一步会新建一批，从原始照片重新生成全部原版和共同中间产物，重置后两步的当前交接指针，保留旧文件用于追溯。重新执行第二步只重新计算当前批次的改进版；重新执行第三步只重新计算评价。不要在同一个输出根目录并发启动不同批次。某一步失败后，后续入口拒绝继续，不自动选择旧成功结果。

## 固定数据调试：modify模式

第一次按上面三个命令顺序执行，每个命令都添加 `--mode modify`。其他输入参数不变，无需填写特征、模型、匹配、配准或缓存文件路径。缓存由本次输入生成，不自动导入历史实验中的模型。

只修改 `joint_map/joint.py` 中的正式联合求解函数、公共准备依赖未变时，后续通常只需要执行：

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_improved.py \
  --output-dir /media/amax/PS2000/pixelsfm/outputs/my_batch_1114 --mode modify

/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/evaluate_results.py \
  --output-dir /media/amax/PS2000/pixelsfm/outputs/my_batch_1114 \
  --gt-poses /home/amax/Documents/code/pixel-perfect-sfm-main/datasets/poses/DJI_20251221132525_0004_V.txt \
  --gt-depth-dir /home/amax/Documents/code/pixel-perfect-sfm-main/datasets/images/DJI_20251221132525_0004_V --mode modify
```

修改了输入、默认参数、地图匹配、公共先验或后处理代码时，先重执行第一步并加 `--mode modify`，再执行后两步。第一步只重新计算失效部分；若跳过第一步，改进版入口会检查公共依赖并提示先更新第一步。

| 变化 | modify处理 |
|---|---|
| 输入、参数、实现均未变 | 校验并复用原版完整产物与共同准备 |
| 换图像或内参 | 重建原版及受影响阶段；仅内参变化时整幅图像特征仍可复用 |
| DOM/DSM变化 | 保留原版重建和整幅特征，重新匹配地图、标定、配准及原版后处理 |
| 公共标定/先验实现变化 | 复用有效原版、地图初始化快照和整幅特征，重做标定与后续阶段 |
| 仅原版稠密/DSM/导出实现或更新参数变化 | 复用共同准备，重新执行原版后处理 |
| 仅正式联合求解函数变化 | 公共准备不失效；每次重新生成改进版联合结果、MVS、DSM和导出 |
| 改进版结果变化，基线与GT未变 | 复用基线反投影参考及基线误差，重新计算改进版误差 |
| GT、基线、相机、原DSM或评价协议/实现变化 | 重新建立评价参考；GT缓存仅供第三步使用 |

依赖指纹包含原始文件内容SHA-256、输入路径、相机参数、相关阶段源码、本地权重及核心原生库。相关源码保守失效，例如修改整个 `optimization.py` 会重做公共标定；新增联合模块的全局变量或导入也可能使公共准备失效。缓存命中仍须读取文件验证摘要，并运行必要的一致性检查，因此耗时不会降为零。改进版MVS、求解和全像素新误差仍需计算；此次未重跑完整真实流程，未测量加速倍数。

`result.json`及评价`report.json`记录 `mode`、`cache_allowed`、`cache_used`、`cache_hits`、`cache_events`和`input_fingerprint`；原版另记录各阶段 `dependency_fingerprints`。终端显示“modify缓存命中：阶段名”。未命中原因记录在`cache_events.reason`中。只有完成的阶段才发布缓存，文件缺失或内容变化会失效；清单JSON格式损坏会报错停止。

开发缓存只存校验清单与小型初始化快照，大文件通过开发输出目录链接复用。保留需要复用的旧 `modify` 目录及 `modify_cache`；删除或改动源文件将导致缓存失效，程序不会自动清理历史数据。初始化快照使用独立副本，后续标定不会覆盖它。

**正式交付使用 `--mode real`（或省略参数）重新运行第一步，然后运行后两步。** `real` 不实例化开发缓存读取器；拒绝`modify`批次、结果及指向开发目录的链接。相同输出根目录切换模式时，第一步会更新当前`batch.json`，后两步的模式必须匹配；需要保留两种模式各自的当前交接时，使用不同输出根目录。已有无模式字段的历史完整结果仍可只读评价；开发模式评价旧结果时，缓存也只写入新的开发评价目录。

## 输入约束与环境

- 六张RGB图像，3840×2160；支持PNG/JPG/JPEG及大写扩展名；文件名不含空白，stem互不重复，按文件名排序。内参由第一步提供并在优化中固定。
- DOM：EPSG:4326，三波段int16，NoData=255；DSM：共网格、至少2×2、单波段float32、NoData=-9999；高度约定WGS84椭球高。地图尺寸、轨迹、观测和先验数量由本批输入决定。
- 使用已有`/opt/anaconda3/envs/pixsfm/bin/python`、本地模型权重和CUDA COLMAP。不会下载或修改库。所有批次产物、缓存和日志从给定输出根目录派生。
- 位姿经纬度和角度单位为度，高度为米。姿态采用相机当地ENU，`pitch=0`表示俯视；`R_render_to_ENU=Rotation.from_euler('xyz',[pitch,roll,yaw],degrees=True)`，CV/渲染轴转换为`diag(1,-1,-1)`。

`result.json`的`status=complete`表示对应部分执行完成；精度是否通过只看第三步报告。此前GT验收四项未达标的历史结论保留，见[MVP_ACCEPTANCE_20261114.md](MVP_ACCEPTANCE_20261114.md)。

API：`run_original(image_dir, dom_path, dsm_path, output_dir, intrinsics, mode='real')`只生成原版及共同标定；`run_improved(output_dir, mode='real')`只生成当前批次改进版；`evaluate_results(output_dir, gt_poses, gt_depth_dir, mode='real')`只计算误差。旧的合并入口`run_localization.py`已移除。

## 缓存模式本次验证记录

执行命令（测试目录已创建）：

```bash
TMPDIR=/media/amax/PS2000/pixelsfm/outputs/modify_mode_validation_20261114/tests_tmp /opt/anaconda3/envs/pixsfm/bin/python -B -m unittest discover -s tests -p 'test*.py' -v
```

```text
Ran 177 tests in 19.610s
OK
```

[完整测试日志](/media/amax/PS2000/pixelsfm/outputs/modify_mode_validation_20261114/all_tests.log)。包含17项新增缓存回归：命中与分阶段失效、失败阶段不发布缓存、模式混用拒绝、开发目录链接拒绝、原版不可变、改进版每次四项重算、GT缓存隔离、实际小型评价参考复用及真实模式重新建立参考。重建缓存测试模拟耗时后端；评价复用测试实际计算148个合成有效像素。现有环境依赖导入及本地权重/源码指纹读取已通过。**此次没有重跑用户六图完整重建和全像素GT评价，也没有更新历史精度结论。**

## 缓存改动之前的三个入口实际验证记录

本机已分别启动并完成三个命令，没有使用合并启动脚本作为交付入口。原版独立完成10个内部步骤后停止；随后单独启动改进版的4个步骤；最后再单独启动全像素评价。

| 核验项 | 实测结果 |
|---|---|
| 原有及新增回归 | 152项通过，38.717秒；包含输入适配、三个入口隔离、旧批次拒绝、评价只读与各项误差反例 |
| 原版边界 | 未执行正式joint，完成时改进版和评价均未启动；六图位姿已导出 |
| 改进版边界 | 仅joint、dense_joint、update_joint、export_joint；63项原版关键产物摘要未变 |
| 评价边界 | 未启动重建；评价后两版119项受保护产物摘要均未变 |
| 全像素参考域 | D=49,766,333，基线先冻结D0=49,766,286；两版有效位姿均为6 |
| 精度结论 | 8项中4项达标；旋转median/std及DSM median/std仍未达标，未用GT回调参数 |

回归命令与输出：

```bash
TMPDIR=/media/amax/PS2000/pixelsfm/outputs/separate_workflows_20261114/tests_tmp /opt/anaconda3/envs/pixsfm/bin/python -B -m unittest discover -s tests -p 'test*.py' -v
```

```text
Ran 152 tests in 38.717s
OK
```

完整记录：[separate_workflow_audit.json](/media/amax/PS2000/pixelsfm/outputs/separate_workflows_20261114/separate_workflow_audit.json)，含三个实际命令、独立输出目录、源码摘要及边界核验。[本次误差报告](/media/amax/PS2000/pixelsfm/outputs/separate_workflows_20261114/batch/evaluation/run_20261114T091316_512221Z/evaluation/report.md)。这里的工程执行通过与精度验收未通过分别记录。
