<!-- 用途：说明仅提供原始图像、地图、输出路径和内参的一条命令定位流程。 -->
# 六张照片与地图：一条命令完成定位

你只需提供图像目录、DOM路径、DSM路径、输出根目录和内参。**不需要配置JSON，不需要原版模型、特征缓存或稠密工作区，也不需要GT或中途确认。** 程序在本次运行中自动生成这些产物。

在终端执行下面这一条命令；把路径和四个内参值换成你的新数据。命令使用绝对入口路径，可在任意工作目录执行：

```bash
/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_localization.py \
  --image-dir /你的新数据/images \
  --dom /你的新数据/dom.tif \
  --dsm /你的新数据/dsm.tif \
  --output-dir /media/amax/PS2000/pixelsfm/outputs/my_batch \
  --intrinsics 2700 2700 1920 1080
```

这是一条命令，反斜杠仅用于换行。四个内参依次为 **fx fy cx cy**，单位为像素；示例数值不是对新相机的默认假设。运行使用现有pixsfm环境和GPU，无需激活或修改环境。

## 只需要设置的参数

| 参数 | 内容 |
|---|---|
| `--image-dir` | 本批六张RGB照片所在目录；支持`.png`、`.jpg`、`.jpeg`及大写扩展名，按文件名排序。 |
| `--dom` | 本批DOM GeoTIFF路径。 |
| `--dsm` | 本批DSM GeoTIFF路径。 |
| `--output-dir` | 输出根目录；示例位于用户指定的输出盘，程序从参数派生路径，不写死数据目录。 |
| `--intrinsics` | 本批固定PINHOLE内参`fx fy cx cy`；fx/fy须为正，四值须有限。优化不改变内参。 |

当前图像协议仍为**六张、RGB、3840×2160**，文件名不含空白，stem互不重复。输入目录中不要混入其他照片。无需手工列出图像名称、配置内部路径或复制旧模型。

DOM须为EPSG:4326、三波段int16、NoData=255；DSM须为单波段float32、NoData=-9999，与DOM共网格，至少2×2像元。高度约定为WGS84椭球高。地图尺寸、仿射变换、图像tracks、观测和地图先验数量由本批输入确定。入口不会静默转换地图坐标系或高程基准。

## 每次都重新生成

每次启动自动创建独立目录：

```text
<output-dir>/run_<UTC时间戳>/
```

即使输入和命令相同，也创建不同目录，重新执行特征提取、匹配、原版重建、地图关联、验证和定位。**不复用上次运行的数据缓存、模型、先验或定位结果。** 运行中生成的缓存仅服务本次内部阶段；现有预训练权重属于算法依赖，不是旧数据缓存。

内部步骤由程序自动执行：

```text
六图与地图检查
  → SuperPoint特征与SuperGlue匹配
  → 原版PixSfM：关键点调整、SfM、特征BA
  → 原版CUDA稠密重建
  → DOM/DSM关联、2 m共同一致性筛选
  → 整幅特征缓存、三候选验证与自动选择
  → 共同地图配准、正式联合优化
  → 两分支稠密处理、可靠性筛选及DSM更新
  → WGS84位姿和点云导出
```

候选选择只使用本次无GT验证：存在明确优势候选时采用该通过者；否则在通过者中按1.0、0.1、10.0顺序选择。没有通过者就停止，不使用旧确认或GT择优。自动选择记录为`automatic_no_gt_v1`，不冒充用户手动选择，也不宣称获得绝对精度最优权重。

## 定位结果在哪里

控制台开头打印本次目录，结束打印位姿文件和结果JSON的绝对路径。以下路径均相对于本次`run_...`目录：

| 文件 | 内容 |
|---|---|
| **`joint/export/poses_wgs84.txt`** | **主要定位结果：六张照片的经纬高和姿态。** |
| `localization_result.json` | 总状态、六图位姿数组、自动选择记录、各步骤退出码、日志和输出路径。 |
| `joint/export/sparse_wgs84.ply` | float64经度/纬度/椭球高稀疏点云，带RGB。 |
| `joint/export/dense_wgs84.ply` | 同坐标约定的稠密点云。 |
| `joint/map/updated_dsm.tif` | 更新后的DSM。 |
| `joint/map/summary.json` | 实际更新数量与更新状态；无可靠更新的位置保留原值。 |
| `native/` | 本次生成的原版特征、匹配、PixSfM模型、缓存、稠密工作区与来源记录。 |
| `shared/`、`cache/tmp/` | 本次地图关联、验证、特征缓存及临时文件。 |
| `logs/localize_*.log` | 各内部步骤的完整stdout/stderr；失败时控制台指明对应日志。 |
| `reconstruction.json` | 自动生成的内部配置；**不需要用户编辑**。 |

位姿文件无表头，每行：

```text
image_name longitude_deg latitude_deg ellipsoidal_height_m roll_deg pitch_deg yaw_deg
```

输出顺序为输入照片的文件名排序，角度单位为度、高度单位为米。姿态采用相机当地ENU，`pitch=0`表示俯视；`R_render_to_ENU=Rotation.from_euler('xyz',[pitch,roll,yaw],degrees=True)`，CV/渲染轴转换为`diag(1,-1,-1)`。内部几何使用地图原点定义的ENU米制坐标，经纬度PLY仅用于地理导出。

## 失败与运行边界

某一步失败后立即停止，不继续使用不完整产物。失败目录保留用于排查；再次执行同一条命令会创建新目录，从原始输入重新生成，不续用失败缓存。

需要现有可访问的CUDA GPU、已有本地SuperPoint/SuperGlue/S2DNet权重及CUDA COLMAP。权重缺失会报错，不下载、不重装或修改库。COLMAP路径和既定算法参数由内部协议提供，普通使用不需要设置这些路径。临时文件和中间产物随本次输出目录生成；原版准备进程的IPC套接字使用该目录下的相对路径，避免长绝对路径超出Linux套接字限制。

`status=complete`表示定位执行完成。入口不要求也不读取GT，结果明确记录`accuracy_evaluated=false`。此前独立GT验收四项精度未达标的结论保留，见[MVP_ACCEPTANCE_20261114.md](MVP_ACCEPTANCE_20261114.md)。本入口的工程验证不能替代新场景绝对精度验证。

`run_joint_map.py --stage ...`仍保留为内部阶段调试接口；正常交付使用本文开头的一条命令。

API：`run_localization(image_dir, dom_path, dsm_path, output_dir, intrinsics)`创建全新运行并返回结果字典，同时写入`localization_result.json`。`check_registered_visual_model(visual, native, registration)`按原生写入/读取归一化核验共同配准交接，拒绝真实的位姿、几何或观测改动。

## 开发验收记录（2026-11-14）

本机实际从六张原始照片和DOM/DSM启动，14个内部步骤全部退出0，自动选择本批通过的1.0候选并生成六行位姿；没有提供原版产物、GT或中途确认。关键中间文件的生成时间晚于本次启动，各路径均归属于本次目录，整幅缓存记录`reused=false`。两分支均有实际DSM更新。该记录是工程流程验收，未进行本轮绝对精度评价。

- 本次目录：`/media/amax/PS2000/pixelsfm/outputs/raw_localization_delivery_20261114/run_20261114T072636_808054Z`。
- 完整命令及核验：[delivery_audit.json](/media/amax/PS2000/pixelsfm/outputs/raw_localization_delivery_20261114/delivery_audit.json)。
- 实际控制台输出：[delivery_attempt4.log](/media/amax/PS2000/pixelsfm/outputs/raw_localization_delivery_20261114/delivery_attempt4.log)。
- 定位文件：[poses_wgs84.txt](/media/amax/PS2000/pixelsfm/outputs/raw_localization_delivery_20261114/run_20261114T072636_808054Z/joint/export/poses_wgs84.txt)。

开发回归命令与实际结果：

```bash
TMPDIR=/media/amax/PS2000/pixelsfm/outputs/quaternion_handoff_fix_20261114/tmp /opt/anaconda3/envs/pixsfm/bin/python -B -m unittest discover -s tests -p 'test*.py' -v
# 入口及原有回归：Ran 147 tests ... OK
TMPDIR=/media/amax/PS2000/pixelsfm/outputs/quaternion_handoff_fix_20261114/tmp /opt/anaconda3/envs/pixsfm/bin/python -B -m unittest discover -s tests -p 'test_joint_map.py' -k TestFormalJointOptimization -v
# 第二处四元数修复后定向复测：Ran 11 tests ... OK
```

入口测试使用独立生成的六张照片和5×7地图，不依赖历史数据集或原版模型；另验证连续启动生成独立目录、不同内参、失败即停和长输出路径下的真实跨进程张量传输。四元数检查分别核对保存前指纹与归一化后的持久化状态，覆盖三个旋转和真实改动拒绝，不放宽数值容差。


/opt/anaconda3/envs/pixsfm/bin/python -B -u /home/amax/Documents/code/pixel-perfect-sfm-main/run_localization.py \
  --image-dir /home/amax/Documents/code/pixel-perfect-sfm-main/datasets/images \
  --dom /home/amax/Documents/code/pixel-perfect-sfm-main/datasets/maps/dom.tif \
  --dsm /home/amax/Documents/code/pixel-perfect-sfm-main/datasets/maps/dsm.tif \
  --output-dir /media/amax/PS2000/pixelsfm/outputs/my_batch_1114 \
  --intrinsics 2700 2700 1920 1080