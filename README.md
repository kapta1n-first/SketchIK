# SketchIK

SketchIK 是由单个持续 Agent 规划、数值 IK 执行的具身操作项目。Agent 根据 RGB-D、点云和实际回执，逐个子任务提交带几何参数的动作草图；程序完成坐标换算、轨迹编译、关节控制和执行监控，再把真实结果交回同一个 Agent。

## 工作流程

1. 程序进行桌面多视角扫描、按需补拍并复位，保存原生 RGB-D、标定和点表。
2. Agent 通过 `submit_scene_requirements` 声明任务对象、交互部位和需要的观测。
3. 程序交付对象点图、场景总图、尺寸、双 RGB 和机器人状态。
4. Agent 通过 `execute_actions` 提交当前子任务的完整草图，程序依次执行动作与路点。
5. 草图完成后继续下一子任务；实际失败时停止未执行部分，由同一 Agent 根据真实停止状态规划恢复。

默认同步。增加 `--async-planning` 后，同一个 Agent 在当前段执行期间，用冻结的已有资料准备下一段完整草图。当前段正常完成且所用坐标、持物参考及预算检查通过后直接接续；当前段失败时作废候选，取消在途规划，再交回 Agent 恢复。终态提议在实际反馈后由 Agent 提交 `finish`。

## 工具与动作

| 阶段 | 工具 |
|---|---|
| 观察 | `submit_scene_requirements` |
| 执行 | `execute_actions`、`list_dir`、`read_text_file`、`read_image`、`get_state`、`refresh_object`、`finish` |
| 异步提前规划 | `draft_next_sketch` |

草图动作包括 `move`、`rotate_roll`、`rotate_pitch`、`rotate_yaw`、`set_gripper` 和 `wait`。一段最多12项动作，每个 `move` 包含完整位姿路点，默认最多6个。夹爪事件作为独立动作提交。

位置使用米，角度使用弧度，四元数使用 xyzw。路点可选已展示的点云或原生图像像素、当前 EEF，再指定世界坐标或目标工具坐标中的偏移。对齐参考支持末端、单侧指垫、两指垫中点，以及具有有效关联的持物底部或中心。对象 ID 标识所用几何，须与实际选点对应。

一次回复中的 `execute_actions`、`refresh_object`、`finish` 各自独占。格式或调用数量不合规则零动作反馈，默认允许连续3次纠正，第4条无效回复结束。格式纠正计入 API 和工具尝试，不占草图段或物理步额度。

## 执行与恢复

程序以真实 IK 编译绝对关节轨迹，实际发送同一份封存制品，逐物理步刷新状态并检查 FK 一致性。位置容差5毫米、朝向容差3度、最大关节误差0.02弧度。轨迹中的多路点由 Agent 规划；当前整任务采用 IK-only，运动收敛不表示已抓牢、放好或任务完成。

T1 对同一绝对目标换数值初值，或在仍持续接近时续动最多40物理步；停滞、接触变化或继续失败则转 T2，由同一 Agent 重画恢复草图。恢复可包含仍需完成的剩余动作，已完成动作和未执行后缀分别记录。

首次目标闭爪使用已保存的前后 RGB-D、实际指垫和夹爪间隙检查局部接触。持物几何独立维护原点集、抓持 epoch、相对手变换及深度支持；关联不足时可使用当前实测像素、指垫或 EEF 规划恢复。

## 点云和观测

原生观测为1024×1024前视与腕部 RGB-D。初始、扫描停止、动作完成或失败时保存关键帧；每个物理步录制20fps前视视频。API 等待期间不重复静止画面。API 图片采用 JPEG95、不缩放，原 PNG、深度、标定和实际发送内容分别归档。

`current_clouds/latest/` 提供当前总图和局部入口。段末冻结已有观测与几何，后台计算点表、绘图；显式读取尚未完成的新版本时才等待。固定入口指向新版本，原 PNG、点表、观测编号和点 ID 保留。旧图片不会自行变成当前图。

刚体保留初始多视角形状和精确成员。有效抓持中按当前 EEF 传播位置；独立移动后，可用当前 RGB-D 重定位旧形状，完整、唯一且深度支持的配准才开放原成员的新坐标。杯子等物体在爪内滑转，可通过 `refresh_object(mode=bind, rebind_held_pose=true)` 显式重绑同次抓持的相对位姿。匹配不足时保留历史形状和不可用状态。

活动结构与未知形状按局部几何处理。只有特征、空间支持和当前深度共同支持的旧成员才更新；历史背景与当前实测点分层呈现。当前局部点图可直接选点，带 G 编号的来源须填写实际 `group_id`。

Agent 按证据选择读图、检查或关联。柜子和抽屉结束前读取对应活动部位的最新局部点图，结合当前双 RGB 核对目标状态。搬运与放置同时考虑持物朝向、沿途净空、整手范围、退出路线及后续子任务。

## 文件和上下文

Agent 只可读取本轮 `agent_public/` 注册资料：

| 路径 | 内容 |
|---|---|
| `usage.md` | 当前操作说明 |
| `scene.json`、`dimensions.json`、`current.json` | 对象、尺寸及当前状态 |
| `current_clouds/latest/` | 当前总图与局部点图入口 |
| `geometry/`、`scene/` | 点云图、精确像素点表与原始点集 |
| `observations/`、`hand/` | 原生观测与手部几何图 |
| `receipts/index.json` | 本轮草图、实际执行和失败回执索引 |
| `acquisition/shown_native_images.json` | 初始观察阶段已展示原图的继承登记 |

操作说明作为系统正文提供，工具 schema 随阶段发送。长会话保留近期完整工具对话、最新状态及必要图像；请求图片窗口按逻辑槽更新，源会话和原图完整归档，历史公开图可按路径读回。初始观察中已实际展示的原图继承到执行阶段，几何时效检查仍独立生效。

## 运行

使用包含仿真依赖和资产的 Python 环境。运行前检查 GPU 空闲；API 配置放在私有 env 文件中。

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  <python> -m sketchik.task_trial \
  --task 0 --seed 0 --init-state 0 \
  --max-episode-steps 9000 --max-turns 30 --max-tool-calls 100 \
  --max-subtasks 20 --max-route-points 6 --completion-diagnostic \
  --image-jpeg-quality 95 --reference-exact-images \
  --a-image-window --a-context-window --api-stream \
  --env-file /absolute/path/to/private.env \
  --gpu-uuid '<GPU UUID>' --output runs/<new-run-name>
```

私有配置项为 `SKETCHIK_BASE_URL`、`SKETCHIK_MODEL`、`SKETCHIK_API_KEY` 和可选 `SKETCHIK_REASONING`。输出上限与预算字段通过 `--api-max-tokens`、`--api-budget-field` 按接口能力设置。

主线固定使用原生工具调用接口及现有 XML 网关通道，不提供 JSON 文本模拟工具调用。面向仅支持 JSON 文本工具协议的 Bytecat/Kimi 接入，使用 `codex/json-bytecat-kimi` 分支。

连接异常或不完整流会保留原仿真、同一会话和请求边界，等待恢复；等待期间物理步保持不变。原进程仍存活时可执行：

```bash
PYTHONPATH=src <python> -m sketchik.request_recovery /absolute/path/to/run resume
```

自动网络重发由获得授权的运行控制程序负责。认证、额度、格式和实际运动失败分别处理。原始耗时、网络故障区间、有效耗时、异步重叠及段后等待单独记录。

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 <python> -m pytest -q
```

## 实验记录

结果中的模型自报、动作收敛和官方评测分别记录。评测真值只用于初始化与评测，操作 Agent 使用公开观测进行定位和判断。逐次 API 尝试保存请求体积、时间、用量及可取得的余额；共享账户窗口差额与独占请求费用分别表述。

- [异步整套实验记录](reports/async74-seed0-20261117/README.md)
- [外部对照复现审计](reports/harness-audit-20261117/README.md)
- [外部对照工具](tools/harness_eval/README.md)
- [修改与验证记录](changes.md)
