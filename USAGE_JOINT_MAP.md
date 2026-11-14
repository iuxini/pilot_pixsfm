<!-- 用途：说明原版、改进版、误差评价三个独立入口及同批自动交接。 -->
# 三个独立入口：原版、改进版、评价

现在按顺序执行三个命令。第一步提供六张照片、DOM/DSM、输出根目录和内参；后两步通过同一个输出根目录自动读取本批交接，不需要手写配置JSON或填写中间文件路径。GT只传给第三步。

每组数据选一个输出根目录，下面统一用 `/media/amax/PS2000/pixelsfm/outputs/my_batch_1114` 举例。三个命令中的该路径保持一致。

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
├── original/run_<UTC时间戳>/
│   ├── result.json
│   ├── reconstruction.json
│   ├── native/                       # 本批从照片生成的原版产物
│   ├── shared/                       # 两版共用的冻结标定、配准和缓存
│   └── baseline/
│       ├── export/poses_wgs84.txt     # 原版地理位姿
│       ├── export/*_wgs84.ply
│       └── map/updated_dsm.tif
├── improved/run_<UTC时间戳>/
│   ├── result.json
│   └── joint/
│       ├── export/poses_wgs84.txt     # 改进版地理位姿
│       ├── export/*_wgs84.ply
│       └── map/updated_dsm.tif
└── evaluation/run_<UTC时间戳>/
    ├── result.json
    └── evaluation/
        ├── report.md                 # 人可读误差报告
        ├── report.json               # 完整精度与验收状态
        ├── pose_errors.csv           # 两版逐图位姿误差
        └── dsm_errors_*.npy           # DSM全参考域误差
```

每个入口会打印自己的本次目录和结果路径，各自日志位于本次目录的`logs/step_*.log`或`logs/evaluate.log`。

重新执行第一步会新建一批，从原始照片重新生成全部原版和共同中间产物，重置后两步的当前交接指针，保留旧文件用于追溯。重新执行第二步只重新计算当前批次的改进版；重新执行第三步只重新计算评价。不要在同一个输出根目录并发启动不同批次。某一步失败后，后续入口拒绝继续，不自动选择旧成功结果。

## 输入约束与环境

- 六张RGB图像，3840×2160；支持PNG/JPG/JPEG及大写扩展名；文件名不含空白，stem互不重复，按文件名排序。内参由第一步提供并在优化中固定。
- DOM：EPSG:4326，三波段int16，NoData=255；DSM：共网格、至少2×2、单波段float32、NoData=-9999；高度约定WGS84椭球高。地图尺寸、轨迹、观测和先验数量由本批输入决定。
- 使用已有`/opt/anaconda3/envs/pixsfm/bin/python`、本地模型权重和CUDA COLMAP。不会下载或修改库。所有批次产物、缓存和日志从给定输出根目录派生。
- 位姿经纬度和角度单位为度，高度为米。姿态采用相机当地ENU，`pitch=0`表示俯视；`R_render_to_ENU=Rotation.from_euler('xyz',[pitch,roll,yaw],degrees=True)`，CV/渲染轴转换为`diag(1,-1,-1)`。

`result.json`的`status=complete`表示对应部分执行完成；精度是否通过只看第三步报告。此前GT验收四项未达标的历史结论保留，见[MVP_ACCEPTANCE_20261114.md](MVP_ACCEPTANCE_20261114.md)。

API：`run_original(image_dir, dom_path, dsm_path, output_dir, intrinsics)`只生成原版及共同标定；`run_improved(output_dir)`只生成当前批次改进版；`evaluate_results(output_dir, gt_poses, gt_depth_dir)`只计算误差。旧的合并入口`run_localization.py`已移除。

## 三个入口的实际验证记录

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
