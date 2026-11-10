# SketchIK

独立的机器人解析动作执行项目。目前接通 **高层会话/工具接口 → LIBERO 真实观测 → 数值动作计划 → IK → 关节轨迹 → 仿真反馈**，之后再接对象级草图的抓取与放置 Grounding。

已实现兼容 Chat Completions 的高层 Agent 会话与独立操作说明。当前独立抓取诊断已接通真实高层请求、SAM3＋Molmo 身份选择、GraspGen 候选、机器人 IK 检查、实体夹爪渲染和 Molmo 位姿选择，并已执行保存的抓取计划；这不等于正常会话已有自动完整抓放技能。SAM3、Molmo2-ER 与 GraspGen 服务常驻。另有 AnyPlace 离线放置候选诊断，尚未接入正常动作或通过实放验证。当前没有 VLA、memory、World Model 或异步规划。不导入 AsyncSketch / HarnessVLA 的执行器、会话、memory 或纠错代码。

## AnyPlace 放置候选诊断

`place_preview` 使用官方 [AnyPlace](https://github.com/ac-rad/anyplace) 源码和公开权重；低层输入为持物可见点云与目的地区域可见点云，输出物体相对当前世界姿态的 SE(3) 变换。它不直接读取任务语言，不负责关门。模型未启用成功分类器，所有候选都须另外核验。夹爪目标为 `T_world_delta @ T_world_current_EEF`，前提是物体相对夹爪刚性保持；不是单纯给目标点加高度。

```bash
PYTHONPATH=src .venv-grasp/bin/python -m sketchik.place_preview \
  --checkpoint .runtime/anyplace_ckpts/anyplace_multitask/model.pth \
  --held-cloud <抓取后SAM/Molmo提取的observed_object.npz> \
  --held-state <同一帧state.json> \
  --destination-observation <目的地双视角RGB-D目录> \
  --destination-point <包含实测world_xyz的选点记录.json> \
  --gpu-uuid <空闲GPU或已批准模型服务所在GPU的UUID> \
  --peer-pid <本项目常驻模型PID> --output runs/<新结果目录>
```

依赖官方源码位于 `.runtime/AnyPlace`，环境另需 yacs、healpy 和已有 torch_cluster/scatter。保留官方推理与严格权重加载；只抑制 Meshcat 展示、将 airobot 日志映射到标准日志。未安装的旧 knn_cuda 分支一旦实际调用会明确失败，不以假计算代替。当前是独立诊断进程，不宣称 AnyPlace 已成为常驻服务。

`--crop-radius` 指定围绕已观测目的地点的局部立方体区域；可选 `--plane-range low high` 进一步按实测局部平面裁剪。这些是诊断输入选择，不能证明整个腔体可通行。下采样只选择真实点，不补全隐藏面。`place_inspect --preview <结果目录> --robot-observation <保存robot_only.xml与metadata.json的目录> --gpu-uuid ... --peer-pid ...` 渲染实体夹爪与变换后的可见物体点云，并检查最终IK、直线段IK和局部平面高度；它不包含机械臂完整避碰、开口识别、撤出规划或释放授权。

2026-11-10 的 task9 诊断共生成80个候选：宽区域20个悬空；窄区域、stacking权重和排除底面以下杂点的另外60个出现底面以下点。宽区域和最后一组共40个完成双视角实体渲染与IK检查，均未通过局部底面间隙检查；未实放。抓取后观测与目的地观察来自两个独立诊断回合，不能直接混合作为在线规划状态。汇总为 `runs/20261110_anyplace_prepare/summary.json`，可见点云图为该目录的 `microwave_cloud.png`，可旋转版本为 `microwave_cloud.html`。这些结果仅说明当前权重/输入组合尚不可用，不足以否定该方法在其原评测条件下的表现。

## 当前动作

| 接口 | 含义 |
| --- | --- |
| `move_to(position, quaternion_xyzw=None)` | 移到世界坐标 XYZ；不填朝向则保持进入动作时的朝向 |
| `move_relative(delta_m, quaternion_xyzw=None)` | 相对进入动作时的真实 EEF 位置移动；XYZ 均可正负，不自动上抬 |
| `move_pose(position, ...)` | 移动并控制完整朝向，使用四元数或下面的两个方向 |
| `move_waypoints(waypoints)` | 按调用方给定的多末端位置顺序移动；全部路点 IK 成功后才开始这一条动作 |
| `set_gripper(command, steps=15)` | `open` 或 `close`，同时保持当前关节目标 |
| `release(steps=15)` | 只在当前位置张爪；不会选落点或认定放置成功 |
| `wait(steps=10)` | 保持当前关节位置和上一条夹爪命令，推进真实物理仿真 |

`LiberoRobot.state()` 返回机器人关节角、末端完整姿态、手指关节位置、夹爪命令、步数和独立的官方 success。`camera(name)` 返回 RGB、米制深度、内参和世界外参，支持 `agentview` 与 `robot0_eye_in_hand`。`back_project(frame, [x,y])` 将一个有真实深度的像素转换成世界坐标，不补全不可见区域。

## 控制方式

1. 从机器人自身的 MJCF 提取独立机械臂与夹爪模型。IK 模型不包含桌面、物体、容器或其他任务 GT。
2. 使用 MuJoCo 正运动学与 SciPy 有界最小二乘求 7 个关节角。目标包括 XYZ 和完整 SO(3) 朝向；以当前真实关节角为种子，优先得到相近解，检查关节限位与解的连续性。
3. 平移按约 2cm、转动按约 8° 分段插值并连续求 IK。每段生成静止到静止的五次关节轨迹，默认命令速度上限 0.6rad/s、加速度上限 1.2rad/s²。
4. 将 **7 个绝对关节角 + 1 个夹爪命令** 发送给 robosuite 的 `JOINT_POSITION` 控制器，`input_type=absolute`、`kp=150`、20Hz。控制器通过关节 PD 与补偿力矩推进物理；没有 OSC，也没有执行时瞬移关节。
5. 每步核对真实后端收到的关节目标。末端完成要求位置误差 ≤5mm、完整朝向误差 ≤3°、最大关节误差 ≤0.02rad；最多有界等待 80 步。未收敛则返回失败，JSON 执行器停止后缀。

IK 解的验收更严格，为 ≤0.5mm / 0.5°；它与真实控制跟踪误差分开记录。命令轨迹的速度限制不代表实际物理速度已经得到安全认证。

**没有自动环境避障、自碰撞检查或物体碰撞模型。** 多路点可表达绕行路线，但路线需要调用方给定；终点可达不等于途中无碰撞。动作执行器还没有持物认证、容器可放区域估计或自动抓放技能；GraspGen 另提供候选位姿及模型评分预览，尚不参与实际动作。

## 参数与朝向

- XYZ、偏移均为世界坐标系，单位米；角度、关节角为弧度；四元数顺序为 `xyzw`。
- EEF 是 Panda 的 `grip_site`，不是物体底部或腕关节。
- `move_pose` 可以填 `quaternion_xyzw`，也可同时填 `finger_direction` 和 `closing_axis`。两个方向都用世界坐标，程序计算完整姿态；Panda `grip_site` 的 +Z 朝向指尖，+X 沿两指分离轴。方向近乎平行则拒绝。
- `move_waypoints` 每个路点只填 `position` 或 `relative_m` 之一，可选 `quaternion_xyzw`。第一点的相对量参考进入该动作时的真实末端位置，后续点参考上一计划路点；朝向未填写则沿用上一计划朝向。
- 移动、调姿、等待都保持上一条夹爪命令，不根据动作名偷偷张爪。张爪/闭爪只是命令，不能证明是否持物。

## 本机运行

```bash
cd /media/wsqlab/more/kaptainiii/projects/SketchIK
./run.sh --help
```

启动前选择实际空闲的 GPU UUID。仿真动作使用 GPU 进行 EGL 渲染，IK 和物理仿真主要在 CPU 上运行；独立 SAM3/GraspGen 服务也在 GPU 上推理。

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
./run.sh smoke --gpu-uuid <空闲GPU的完整UUID>
./run.sh execute --task 9 --seed 0 --init-state 0 \
  --gpu-uuid <空闲GPU的完整UUID> --plan examples/basic_motion.json
```

每次运行先检查该卡的计算进程、显存和利用率，通过 GPU UUID 映射到真实 EGL 设备；不能确认就拒绝启动，不静默换卡。不启动任何常驻模型。

目前**已实测 LIBERO-Pro 的 `libero_10`**。`--variant standard` 预留给安装完整标准 LIBERO 依赖的环境；本机安装的标准包缺失 assets，不能宣称已跑通。它会明确报错，不混用 Pro 的场景资产。

代码独立，当前解释器暂时只读复用旧环境已经安装的第三方依赖与资产：

```text
/media/wsqlab/more/kaptainiii/projects/AsyncSketch/.venv-rpent/bin/python
```

这个路径只存放于被 Git 忽略的 `.runtime/python`；不复制权重或整个环境，也不调用旧项目代码。因此删除这个旧环境会影响本机启动，代码本身没有绑到旧项目执行逻辑。之后可将 `.runtime/python` 改为独立环境解释器，或设置 `SKETCHIK_PYTHON=/绝对路径/python`。`pyproject.toml` 声明通用依赖；**仅 `pip install -e .` 不会自动获得 LIBERO 场景与 init_states**，还需安装兼容的 LIBERO 数据包。

当前仿真环境为 MuJoCo 3.3.0、robosuite 1.5.2、NumPy 1.26.4、SciPy 1.17.1，LIBERO-Pro 安装版本 `rpent-liberopro 0.2.0`，其安装元数据源为 RLinf/LIBERO-PRO 的 `d1e11fb181b8544487d27742c0caa3a4d46452ad`。实际依赖版本和来源记录在每轮 metadata。

所有配置写入本项目 `.runtime/libero_pro`，不修改用户全局 LIBERO 配置。适配层局部绕过已安装 LIBERO-Pro 在 `use_object_obs=False` 时的观测构造错误，只建立机器人观测，不修改第三方源码。

## 输入与日志

`examples/basic_motion.json` 是手工填写、可直接运行的冻结基础动作测试，**不是抓放任务方案，也不是 Agent 生成结果**。JSON 使用 `version=1` 和 `steps`；每步有 `primitive` 与 `params`。`execute` 命令不调用 API；`agent` 命令则根据明确配置调用模型，由模型通过工具提供动作参数。

`seed` 和 `init-state` 分开记录，不把 seed0 等同于全部初始化参数。每次创建环境前重新设随机种子，显式载入指定官方 init_state；载入后只用动作推进物理。没有自动 reset 或失败回滚。

每轮输出在 `runs/<时间>_<命令>/`：

- `metadata.json`：任务、种子、初始状态编号、真实控制器、依赖版本、GPU/EGL 映射与初始化墙钟时间。
- `events.jsonl`：逐步真实关节/末端状态、发出的关节目标、夹爪命令、仿真与视频耗时，及每次 IK/运动结果。
- `result.json`、`video.mp4`：控制验收结果与 20Hz 动作视频。仿真秒数与墙钟秒数分开，加载时间不伪装成动作时间。
- 初始与动作边界的双视角 PNG、深度/标定 NPZ、机器人状态。`robot_only.xml` 用于复核 FK/IK 模型。

## 本次真实验收（2026-11-08）

| 环境与运行 | 结果 | 动作数 | 仿真时间 | 执行与记录墙钟时间 |
| --- | --- | --- | --- | --- |
| Pro task0 / seed0 / init_state0，`runs/20261108T180605_smoke` | 4 次运动全部收敛，夹爪开合与不可达拒绝通过 | 177 | 8.85s | 4.87s |
| Pro task9 / seed0 / init_state0，`runs/20261108T182721_execute` | 冻结基础动作 JSON 成功执行 | 90 | 4.50s | 1.99s |

这些耗时不含环境构造与加载（另见 metadata），包括 IK 规划、动作、日志和视频；仿真可以快于真实时间，不代表真实机器人达到同样墙钟速度。task0 四次最终位置误差为 3.45 / 3.18 / 0.17 / 4.55mm，朝向误差最大约 0.84°。夹爪实测关节分离量由约 2.66mm 变为 77.34mm。

同一个 Pro task0 的三次新建环境初始 agentview PNG SHA256 完全一致；这只是本次同配置重复检查，不推广为所有任务随机源均已验证。两项的官方任务 success 均为 false，因为这里只做空中基础运动，没有执行任务抓放。

CPU 独立机器人测试 **8 passed**：完整姿态 IK/FK、不可达与关节限位拒绝、关节轨迹速度/加速度、方向构姿及米制深度反投影。运行方式：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  "$(cat .runtime/python)" -m pytest -q tests
```

下一阶段可以在这个底座上分别接入 Grounding、抓取候选生成与放置目标生成；当前不会把可见表面点直接包装成可靠抓取或放置技能。

## 高层 Agent 与操作说明（2026-11-09）

| 文件 | 用途 |
| --- | --- |
| `prompts/system.md` | 真正发送给模型的完整操作说明：坐标、朝向、动作语义、观测、反馈与恢复。没有任务特定路线和 seed0 配方 |
| `src/sketchik/tools.py` | 模型能调用的工具名称、JSON 参数 schema、参数校验与实际派发 |
| `src/sketchik/agent.py` | API 客户端、图像/状态/工具回执的会话历史、调用预算、完整会话记录 |
| `src/sketchik/cli.py` | `agent` 和 `agent-smoke` 启动入口 |

AGENTS.md 面向修改代码的 Codex，不当作机器人高层提示词。在线模型的名称、URL、reasoning 设置均来自启动参数/配置；没有偷偷选定默认模型，暂未复制旧项目的 API 密钥或经验。

普通高层会话共有 **18 个工具**：7 个基础动作，`observe`、`get_state`、`back_project`、`look`、`segment`、`execute_plan`、`finish`，以及 `read_text_file`、`write_text_file`、`list_dir`、`read_image`。动作底层仍是之前的七个入口；没有新增运动控制器。

`look` 支持当前/历史双视角、原图像素网格、`crop_xyxy`裁剪、1～4倍显示放大和SAM候选标注。显示像素换回原图为`[x/zoom+x0, y/zoom+y0]`；`back_project`始终用原图像素和对应的`observation_id`。`segment(observation_id,prompt)`调用常驻SAM3返回全部候选ID、评分、框和编号Mask图，可分整物或可见部件，不取最高分、不做左右排序。相同观测和提示缓存复用，新观测重新查询。服务地址默认`http://127.0.0.1:18761`，可通过进程环境变量`SKETCHIK_SAM3_URL`指定；工具只连接已启动的服务，不自动启动或重载GPU模型。

本轮不开放`depth`或`measure`，不新增中心/尺寸/建议夹合方向计算。深度和标定由程序内部用于像素反投影，模型无需自己写几何公式。候选的分割分数、唯一候选和合法XYZ都不认证对象身份、抓取或碰撞安全。

文件工具可读 README.md、changes.md、prompts/、examples/ 和本轮公开 observations/，临时计划只能写入本轮 workspace/，已有文件覆盖需显式 overwrite=true。不会开放 API 密钥、评测结果 GT 或其他项目。read_image 使用 observation_id＋camera 读取本轮已有历史 RGB，真实图像重新进入模型上下文，保留原始拍摄步数；不重新拍照、不推进物理。这些是普通运行工具，不是长期经验/memory 系统。

### 每次实际传入模型什么

首次请求：本项目 system 说明、工具 schema、官方任务文字、标注相机名/图像尺寸/source_step 的前视与腕部 RGB、当前关节/末端完整姿态/手指状态/夹爪命令、工具朝向轴、剩余动作额度，以及相机标定。

后续请求：保留完整会话历史，加入模型上一轮调用、真实工具回执。每次基础动作或 execute_plan 返回后，再加入最新双视角 RGB 与机器人状态。只读 get_state/back_project 不额外渲染；observe 可以主动获取新图。**不逐仿真步请求模型，也不要求每个 primitive 都经过一次模型检查**；execute_plan 可以让多个动作在本地连续执行，出错即停止后缀。

深度图保存在本地。高层通过 `back_project(observation_id, camera, pixel_xy)` 获取该观测像素的米制世界坐标；这是可见表面点，不是自动抓取点。历史测量返回原始步数和 current 标志，不静默伪装成新观测。官方 success 不发给模型作 oracle，只留在本地评测结果中。

职责上，高层根据 RGB 判断语义/选择像素，执行器用深度、内外参求 XYZ。目前观测文本仍包含标定矩阵；这不是高层每轮规划的必要输入，本轮文件工具改动没有改变原观测格式。不要把“执行器有深度与标定”误说成“高层直接读深度图或需要自行反投影”。

### 参数和回执

例如模型调用 `move_relative`，参数为：

```json
{"source_step": 0, "params": {"delta_m": [0.0, 0.0, 0.02]}}
```

程序从 step0 的真实位姿上移 2cm，回执包含是否收敛、实际位置/朝向误差、动作步数、规划/执行耗时、最新状态。外层 ok 表示请求已处理，内层 converged 才是运动是否达到容差；两者都不能证明抓到物体或任务完成。

模型也能调用 execute_plan 提交多个动作；参数同样由模型生成，不要求人手写 JSON。当前仍是**数值解析动作计划**，对象名字到抓放位姿的自动求解尚未接入。

动作参数中的 source_step 需对应当前状态，以免执行基于旧画面计算的目标。一次回复中如果提交多个独立工具，按顺序执行；前一动作改变了环境步数后，旧 source_step 会被拒绝。多步连续动作应使用 execute_plan，或等新回执后继续调用。没有每子任务只能规划/恢复一次的规则。

未收敛会返回当前状态供高层调整，不自动重播已执行前缀。参数错误、不可达目标可修正后继续；未知/异常执行停止会话。模型回复截断、工具调用 ID 重复或 HTTP 失败不执行隐式重试。API 总请求耗时与工具耗时分别记录；当前没有流式 TTFT 指标。

### 启动与验收边界

不联网的接口验收（脚本模拟回复，真实 LIBERO/IK 执行）：

```bash
./run.sh agent-smoke --gpu-uuid <空闲GPU的完整UUID>
```

真实模型需要同时支持图像输入与 Chat Completions function tools。填写本地配置 `.runtime/agent.env`，字段见 `.env.example`，不要提交密钥：

```bash
./run.sh agent --env-file .runtime/agent.env --task 0 --seed 0 --init-state 0 \
  --gpu-uuid <空闲GPU的完整UUID>
```

只有显式运行 agent 才会请求所配置的 API，可能产生费用。没有自动网络重试；reasoning_effort 原样传给网关，不宣称网关支持任意值或会关闭推理。兼容格式依据 [OpenAI Chat Completions 文档](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) 与 [function calling 文档](https://developers.openai.com/api/docs/guides/function-calling)；第三方网关兼容性仍需独立实测。

会话目录除原有日志外，包含最终 system_prompt.md、tools.json、无密钥的 agent_config.json、transcript.json 和 observations/。transcript 中的图片使用哈希引用，原始像素/深度/标定保存在 observations；内存中的会话请求保留完整图片历史，不静默压缩。

本轮 `runs/20261109T002029_agent-smoke` 已通过：真实 Pro task0 中完成上移 2cm＋wait，共 18 个物理步；旧 source_step 动作被拒绝且没有额外移动，之后成功结束会话。总墙钟约 1.55s，官方任务 success=false；`real_model_used=false`。本轮未发收费模型请求，不能据此宣称真实 Agent 规划或抓放已通过。CPU 测试累计 **18 passed**，其中包含本地 HTTP 请求格式、图像/状态打包、历史、重复 ID 拒绝、失败恢复、预算与旧观测拒绝。

随后补齐四个文件/历史图片工具，CPU 测试累计 **27 passed**，新增验证文件创建/读取/列目录/覆盖、指南只读、路径与软链接约束，以及历史 RGB 确实进入模型多模态请求且不重新拍照。新增文件工具未额外运行 GPU 或收费模型实验；原真实仿真记录对应此前 12 工具版本，不能冒充新工具的在线模型验证。

## 与原版 HarnessVLA 的实际差别

对照对象为独立 projects/harnessvla 仓库的原版在线运行链路；不是我们先前改造的 AsyncSketch。

| 类别 | 原版已有的能力 | SketchIK 当前状态与实际影响 |
| --- | --- | --- |
| 已有基本等价入口 | 模型调用、图像/状态、工具参数、执行回执、操作指南、文件/历史图片读取 | 已有基础实现；真实模型尚未联调，不能等同于原版完整运行验收 |
| 输入配置有差别 | view_env_state 可附 policy 图、1024 前视图、1024 腕部图；前视和腕部来自两个相机，不是三个相机 | 当前固定原生 1024 前视与腕部 RGB-D，不提供 256 policy 图；已实际渲染双视角与批量分割，但尚未做高层 API 同条件评测 |
| 尚缺工程能力 | 长会话压缩 | 当前保留完整历史；长任务可能超出模型上下文 |
| 尚缺工程能力 | 多种 Planner 后端（API、Codex、Claude Code 等） | 当前只有 Chat Completions 兼容客户端，不提供这些独立 Planner 接入 |
| 尚缺工程能力 | 环境服务/RPC、Dashboard 和取消/交互等运行管理 | 当前主要单进程同步 CLI，不具备原版整套服务与交互管理 |
| 尚缺工程能力 | 更完整的评测、经验导出与 replay 等流水线 | 当前只有单轮执行和证据记录，没有整套批量评测流水线 |
| 用户暂缓 | memory/经验管理 | 暂不接；本轮 workspace 文件不等于长期经验库 |
| 主动改变方案 | 原版解析运动＋冻结 VLA；有 SAM3 服务与分割工具 | 当前 IK＋关节控制，没有 VLA；另有 SAM3＋GraspGen 候选预览，尚未接入高层动作会话。不用 VLA 是路线选择，不是少写一个文件夹 |

对象级草图、执行时自动 Grounding、三层恢复和异步规划属于本项目的研究设计/之前 AsyncSketch 的新增机制，不能算作 HarnessVLA 原版已具备而新项目漏实现的模块。当前 execute_plan 只是数值动作计划，不能冒充对象级草图。独立的新项目尚未迁移这整套机制。

同样，稳定的自动抓取/放置位姿生成、抓取评分和完整场景避障不能未经源码/实验依据就列为 HarnessVLA 已独立解决的模块。原版借助 Agent、几何工具和 VLA 完成操作；我们不用 VLA 后，需要自行补的操作能力，应与复刻框架的工程差距分开讨论。


## 常驻双视角分割与抓取候选预览（2026-11-09）

`src/sketchik/perception_service.py` 是两个独立的本地 HTTP 服务。模型各加载一次，后续请求复用实例，不在每次 primitive 前重新加载。SAM3 返回全部超过 0.3 阈值的候选 Mask、检测框与分数；Mask 以无损 packbits/base64 传输，可恢复为与输入 RGB 同尺寸的布尔数组。没有在服务端取最高分。按用户最新授权，当前预览每视角取 SAM3 分数最高的候选，也支持显式 ID 覆盖；记录选择依据和分数。最高分不能认证物体身份，两视角也可能各自选错，因此尚不能宣称跨视角匹配可靠。此预览没有接入 Molmo，也没有自动执行抓取。

完整链路：同一机器人状态下原生渲染的两张 1024 RGB-D → 一次双图批量 SAM3 → 各相机 Mask → 各自 Mask 内所有有效深度像素 → 各自内外参反投影 → 在世界坐标系拼接可见点云 → GraspGen 的 Panda diffusion-discriminator → 128 个 4×4 位姿及模型评分。这里保留点云，不用中位数将物体压成一个点。GraspGen 内部按官方逻辑清理离群点并采样；输入不包含仿真物体 GT。模型评分不是实际成功率，也不代表可达或无碰撞。

当前两个服务在 GPU2（UUID `GPU-d3ae1b42-968e-7d78-e774-7626c1233431`）常驻：SAM3 `127.0.0.1:18761`，GraspGen `127.0.0.1:18760`。`GET /health` 查看实际 PID、加载耗时和请求计数，`POST /infer` 执行推理。SAM3 输入为 `{"prompt":"目标描述","images":[{"camera":"agentview","image_base64":"…"},{"camera":"robot0_eye_in_hand","image_base64":"…"}]}`；旧单图 image_base64 输入已移除。图像编码器和 grounding 均按 batch=2 调用，不是拼图，也不是连续发送两个单图请求。模型内部仍按官方 1008 分辨率处理，返回 Mask 对齐各自原始 1024 图。PID 与启动参数保存在 `.runtime/perception/`、`.runtime/*_launch.json`；日志在 `.runtime/sam3_service.log` 和 `.runtime/graspgen_service.log`。服务独立于本次终端，但尚未配置系统开机自启或崩溃自启。重启前必须重新检查 GPU，`--peer-pid` 仅允许同项目、同用户的已驻留模型服务；不会停止别人的进程。

首次部署使用 `.venv-grasp`，通过只读 pth 复用已有第三方 torch 2.7.1+cu126 等基础包，其他兼容依赖隔离安装；没有修改旧项目环境。PointNet2 扩展以 CUDA12.3 编译。GraspGen 使用 diffusers0.11.1、huggingface-hub0.25.2、transformers4.44.2；无需 FlashAttention。官方可信 checkpoint 加载显式沿用上游 torch.load 的非 weights-only 语义。SAM3 沿用已安装第三方包和只读权重。上游源码及模型留在 `.runtime`，不提交权重或环境到 Git。

来源：[NVlabs/GraspGen](https://github.com/NVlabs/GraspGen)，源码提交 `2dd8852e1be60f5f9d277fafcc621835cdf59110`；Panda 权重与样例来自 [adithyamurali/GraspGenModels](https://huggingface.co/adithyamurali/GraspGenModels)。当前明确调用原始 diffusion-discriminator，不是上游 demo 新默认的 GraspMoE/OBB。源码没有修改，扩展编译产物除外。权重校验值、包版本与本次服务状态保存在 `.runtime/graspgen_setup.json`。

生成可旋转的三维预览与 PNG：

```bash
USE_FLAX=0 USE_TF=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv-grasp/bin/python -m sketchik.grasp_preview \
  --observation-dir runs/20261109T021024_probe \
  --target 'a cylindrical can with a red and green label'
```

也支持 `--sample-json` 读取官方真实点云样例，仅读取 pc/颜色，再实时推理，绝不拿样例存储的抓取结果冒充模型输出。每次新建运行目录，保存原图、完整 SAM 回应、Mask 叠图、选中 Mask、标定点云、原始候选/评分、三视角线框图、前三个 Panda 实体夹爪渲染、交互式 HTML 和耗时。例如 `--candidate-id agentview=1 --candidate-id robot0_eye_in_hand=1` 可显式覆盖各相机的最高分选择。全部原始点云保留并记录相机来源；当前直接拼接，不去重，也不补全不可见部分。此程序不推进机器人，不调用 IK，不执行抓取。

先前单张 256 图的真实 GPU 结果（历史对照，不是当前默认输入）：

| 输入与阶段 | 结果 | 推理墙钟 |
| --- | --- | --- |
| 官方真实点云样例，GraspGen 首次推理 | 128 个候选 | 5.408s |
| 保存的 LIBERO task0 前视 RGB，SAM3 首次成功推理 | 1 个 Mask，777 个有效三维点 | 0.898s |
| 该 LIBERO 点云，GraspGen 已暖机 | 128 个候选 | 0.161s |
| 同一保存输入重复三次，SAM3 | 同一 PID，均返回 1 个候选 | 0.221 / 0.198 / 0.187s |
| 同一点云重复三次，GraspGen | 同一 PID，均返回 128 个候选 | 0.172 / 0.143 / 0.144s |

这些是此前单张 256 RGB、单物体、128 抓取候选的推理耗时，不含模型加载和绘图，不能推广为完整机器人任务耗时。两服务加载后 GPU 总占用约 6.6GiB（含缓存和 CUDA 上下文）；torch 单进程峰值 allocated 为 SAM3 5088MiB、GraspGen 737MiB，不能把它与 nvidia-smi 总占用混为同一指标。

LIBERO 证据：`runs/20261109T015914_grasp_preview`，含 `metadata.json`、`result.json` 和 `resident_repeats.json`。官方样例证据：`runs/20261109T015805_grasp_preview`。首次 SAM 请求曾因 BF16 转 NumPy 报错，已修为 float32 后序列化并重启本项目服务，错误日志保留。最终 29 项 CPU 测试通过，包含所有有效 Mask 像素的三维转换和多候选不默选最高分。

**本轮只验证生成与显示。** 部分高分罐子候选从下方接近，可能与桌面碰撞；没有场景碰撞筛选，不能直接执行。GraspGen 输出是其夹爪基座约定，后续必须对齐本机器人 grip_site/TCP，检查 IK、关节与场景碰撞，再验证真实抓取。渲染图不是已成功抓住的画面。


当前默认观测已改为前视、腕部各 1024×1024 RGB-D；CLI 不再提供 --size 选项，也没有额外 256 policy 图。旧 runs 中的 256 证据保留，双图预览会拒绝旧 256 图，绝不上采样冒充原生高分辨率。高层仍收到带相机标签的两张图和各自标定。单张看不到目标时记录无检测，只使用另一视角的实际点，不伪造 Mask；两个视角均无检测则失败。

当前双图实测（Pro task0 / seed0 / init_state0，机器人动作 0）：

| 环节 | 实际结果 | 推理墙钟 |
| --- | --- | --- |
| 原生双视角渲染 | 两张 1024 RGB＋各自1024深度、内外参，未使用旧图放大 | 环境初始化 10.94s；不是分割耗时 |
| SAM3 双图首个请求 | 前视2候选、腕部3候选，全部保留 | 1.260s |
| SAM3 双图后续请求 | 同一实例、相同原生双图 | 0.451 / 0.465s |
| 各视角最高分选择＋深度反投影 | 前视 ID1 12556 点，腕部 ID1 13885 点；共26441点 | 见 perception.json 的模型与客户端耗时 |
| GraspGen，26441原始点＋128候选 | 两次均实时生成128候选 | 1.673 / 1.634s |

双图点云比先前777点大得多，GraspGen 官方离群点处理等开销随之增加，不能沿用之前0.16s作为新条件速度。分割/GraspGen时间不含拍照、存图和渲染预览，实际完整程序还有对应开销。两模型仍常驻 GPU2，本轮结束 nvidia-smi 总占用约10.5GiB，含CUDA缓存。

当前最高分规则的完整证据在 `runs/20261109T021343_grasp_preview`：sam_response.json、perception.json、metadata.json、result.json、两相机各自的输入/Mask/点云，以及合并点云、全部抓取位姿和可旋转 HTML。原生图来源为 `runs/20261109T021024_probe`。前一份 `runs/20261109T021246_grasp_preview` 显式指定了相同ID，不能混称为最高分自动选择。

渲染进程可以通过明确的 --peer-pid 与本项目常驻模型共用 GPU；会核验进程所属用户、项目目录和启动模块，任何未授权占用进程仍阻止启动。没有停止他人进程，也没有修改旧项目。CPU 验证33项通过；包含各视角独立标定、响应顺序变化、最高分选择/ID覆盖、空检测不造点和拒绝256输入。真实抓取、场景碰撞和跨视角同物体认证仍未验证。


## 全场景点云与128候选预筛（2026-11-09）

新增独立只读诊断 `src/sketchik/grasp_screen.py`，输入已经保存的双视角原生 RGB-D、选中目标 Mask、原始 GraspGen 候选，以及同次观测保存的机器人状态/robot-only MJCF。不会重新请求模型，不推进仿真，不访问物体 GT，也不修改原始128候选。现有 SAM3/GraspGen 服务继续常驻。

```bash
CUDA_VISIBLE_DEVICES='' USE_FLAX=0 USE_TF=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv-grasp/bin/python -m sketchik.grasp_screen \
  --preview runs/20261109T021343_grasp_preview
```

处理方式：

- 从两个相机提取全部有效可见点（深度0.02～3m），分别按本相机内外参转换到世界坐标，保存每个点的颜色、相机来源和是否属于目标。原始点云全部保存；只有绘图按固定随机种子抽取最多60000点，不把绘图降采样用于碰撞筛选。这是两路可见表面，不是对物体背面、桌子内部等隐藏几何的补全扫描。
- 障碍点云排除目标自身Mask，但保留桌面、篮子、杂物及当前可见机器人。使用全部障碍点建立 CPU cKDTree，按阈值查询，不构造巨大的GPU全距离矩阵。
- 使用机器人自身 MJCF 中完全张开的夹爪碰撞几何（含指垫），采样6000表面点并补充全部154个顶点。检查候选最终夹爪位置，以及沿其自身负Z方向后退10cm的接近段，每1cm一个位姿。任一样本距障碍点小于5mm则拒绝，并记录首次命中前的检查距离；null表示没有样本对落入5mm范围，不是无限净空。
- 10cm、1cm、5mm是本次可调整的离线检查参数，不是强制实际运动路线或任务坐标。CLI提供 --approach-m 和 --clearance-m；它们不改现有基础动作。
- 没有命中上述检查的候选，再求接近起点及最后10cm直线接近的完整姿态 IK；保留现有关节限位、0.5mm/0.5度解验收及分支连续性检查。GraspGen输出是夹爪基座，不直接当 grip_site：从机器人模型实测基座到 grip_site 的变换为Z向0.097m，验证指尖+Z与夹合X轴，转换后再求IK。不能拿GraspGen默认深度0.105m直接替代本机器人站点。

本轮使用同一份原始128候选（源文件SHA256存入result），真实结果：

| 项目 | 数量／墙钟 |
| --- | --- |
| 双视角完整可见点云 | 2097152点 |
| 排除目标自身后的障碍点云 | 2070711点 |
| 夹爪／接近段5mm近距离筛选拒绝 | 119个 |
| 剩余9个中的IK失败 | 2个 |
| 暂时保留 | 7个：原排名28、99、116、117、119、123、128 |
| 点云提取、保存与建树 | 2.84s |
| 全部候选的场景近距离检查 | 7.99s |
| 未被场景检查拒绝项的IK检查 | 4.21s |
| 筛选总墙钟，不含绘图 | 16.17s |

保留项中原第28名评分0.743，为本次最高；其余包括低分或看起来可能夹空的候选。没有用宽松评分门槛把7个说成可用抓法，也未重新采样凑足通过项。原排名前27均被拒绝。

证据目录 `runs/20261109T022431_grasp_screen`：完整点云 full_observed_scene.npz、128项逐条 candidate_checks.jsonl、保持原顺序的 screened_grasps.npz 与retained布尔数组、实测原生夹爪mesh、结果统计、全部候选筛选图、前三个保留项实体夹爪图和可旋转HTML。红色为场景近距离拒绝、紫色为IK拒绝、绿色为暂留。本次几何和IK筛选在CPU完成；前面的候选确由GPU GraspGen生成，两个概念分开记录。

初版无阈值最近邻查询过慢，在没有筛完时主动中断，证据保留在 `runs/20261109T022310_grasp_screen`，没有结果文件，不能当成完成轮次。改成阈值有界查询及遇到拒绝即停止后，完整跑完128项。36项CPU测试通过，其中新增验证接近方向、终点安全但接近段碰撞，以及阈值查询与直接距离计算的拒绝结论一致。

**暂留不等于能执行或能抓住。** 当前只是稠密观测点对夹爪采样表面的近距离筛选，仍可能漏掉遮挡区域、物体内部穿透、采样间隙和完整机械臂碰撞；也没有从当前机械臂到接近起点的完整无碰撞路径规划、自碰撞检查或抓握接触认证。本轮机器人动作0，不能将IK成功或绿色候选解释为实际抓取成功。


## 实体夹爪渲染（2026-11-09）

`src/sketchik/grasp_render.py` 将筛后保留的原始位姿直接用于原生 Panda 视觉mesh渲染：灰色手掌、蓝/橙色两指、黑色指垫；箭头仅示意夹合方向。它不修改候选、不执行动作、不请求模型，也不模拟闭爪接触。对象和局部桌面仍来自真实RGB-D可见点，不用隐藏物体mesh补形。

```bash
USE_FLAX=0 USE_TF=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv-grasp/bin/python -m sketchik.grasp_render \
  --screen runs/20261109T022431_grasp_screen \
  --gpu-uuid <GPU_UUID> --peer-pid <SAM3_PID> --peer-pid <GraspGen_PID>
```

需要本项目 `.venv-grasp` 中的 pyrender0.1.45、pyglet1.5.31、freetype-py；EGL显式绑定GPU，沿用本项目模型同卡验证。只读复用机器人原生mesh，未改其他环境。

实际输出 `runs/20261109T023521_grasp_render`：7个保留位姿的实体近景、完整拼图、原第28名沿最后10cm接近方向平移的42帧/3.5s MP4及GIF，以及将实体夹爪按相机标定投影回前视/腕部原照片的放大图（使用真实深度处理遮挡）。箭头和颜色为辅助标注；照片放大只是显示缩放，不改变输入观测。约9.8s生成全部输出，robot_actions=0。GPU渲染及视频帧数已实测并目视检查，尚未执行抓取。渲染也能看出部分低分保留项可能夹空，因此保留列表不能被称为已认证抓法。


单帧耗时拆分实测：`runs/20261109T024416_grasp_render/render_metadata.json` 新增 render_frame_wall_s。当前3090上，1024照片叠加渲染第一帧0.054s、第二帧0.0167s；900×760候选实体渲染第一帧0.043s、后六帧中位0.0112s；动画42帧的后续中位0.0124s。计时包含GPU渲染与读回CPU，不含准备相机/位姿、文字标注、PNG/GIF/MP4编码及存盘。全部演示资产生成9.04s，独立命令从进程启动到结束11.73s；不能将暖机单帧时间等同于完整工具耗时。渲染器当前仍为按命令创建/销毁，尚未实现常驻渲染或高层候选ID选择回合。候选选择若要接Agent，需另测图像输入、API决策和总闭环延迟。


## 双 Agent 放置预分析小试验

`python -m sketchik.anticipate_trial run --observation <保存的初始观测> --env-file .runtime/agent.env --gpu-uuid <UUID> --peer-pid <SAM_PID> --peer-pid <Molmo_PID> --peer-pid <GraspGen_PID> --history-interval-s 1`先运行A的真实初始观察，再提交符号抓取草图。角色分工如下：

| 角色 | 当前实际职责 | 实际说明 |
| --- | --- | --- |
| A | 看任务和双视角图、决定抓取意图；可以适量分割/反投影辅助空爪观察移动，主要取得新视角和保存历史；不在同一画面反复求精细几何。看得足够时可直接交草图，没有固定次数或路线 | `prompts/initial_observation.md`，拼接`grasp_trial.md`公共抓取说明 |
| 执行器与常驻模型 | SAM3生成Mask候选，Molmo选Mask；GraspGen生成抓取位姿，经原有筛选/渲染后Molmo选位姿或NONE，IK执行 | `prompts/mask_selector.md`、`prompts/grasp_selector.md`及实际工具schema |
| B | A交稿后，与抓取链并行；只读A交接的历史，详细查询后续放置的支撑/入口/接近参考；不接管动作、不替Molmo选抓法，也看不到实时抓取结果 | `prompts/placement_anticipation.md` |

A开放六个感知工具`observe/get_state/read_image/look/segment/back_project`和`move_to/move_pose/move_relative/move_waypoints/wait`。观察移动保持空爪/张开，不开放抓取开合工具；没有自动避障或固定观察动作。最终`submit_grasp_sketch`仍提交位姿引用，由后续候选链绑定，和观察时直接调用的数值运动参数不同。B只有六个只读感知工具；读取的是历史，不会因调用`observe`得到实时图。

历史默认**每20个物理步，即1秒仿真时间**保存一组原生1024双视角RGB-D，另存初始图和每次动作结束/失败的关键帧。`--history-interval-s`可调整，按20Hz量化到物理步。每组保存自己的`observation_id/source_step/标定`，全部周期帧打包给B，不静默丢弃；深度留在本地供感知工具查询。API等待不推进物理，不重复存同一静止画面，也不因此触发模型调用。实际间隔、周期帧ID和控制步数保存在A的`history_manifest.json`，图像及标定在`observations/`，实际A输入/工具/说明单独保存可核对。

当前候选诊断在A观察后关闭环境；候选选定后，新环境重放记录的机器人命令，再严格核对当前RGB-D、标定和状态后实抓。重放不是第二次让A决策，也不恢复物体GT；它是当前诊断入口的额外开销，不能混作正式在线延迟。可选`--lift-test-m 0.05`仍是单独记录的抬升验收，不改模型原计划。

B输出可能的放置/支撑/入口/接近参考像素及用途，程序用同一保存RGB-D反投影为世界XYZ，保存在`prepared_geometry.json`，不能由B猜数。没有可辨认点可返回空；无效深度/越界点单列拒绝。语义用途仍是模型判断，点不等于可释放位姿。参考供以后执行到放置时补全A草图；当前A入口仍只有抓取草图，实际放置步骤、结合抓后状态换算EEF及执行绑定尚未接通。**本节的A主动观察是最新分工；下列较早记录分别按当时权限保存，不能当成已实测当前分工。**

本小试验默认每个Agent最多12轮/60次工具，可用`--max-turns/--max-tool-calls`调整，不影响普通会话上限。API失败不自动重问，不因B失败重放抓取。结果目录`*_two_agent_trial`关联A原试验、B完整历史与真实抓取视频；分别记录B请求开始/结束和物理动作时间，报告实际时间交叠。交叠时长不是已证明的提速，也不表示放置已解决。没有候选时不强选、不启动实抓。

最新实测`runs/20261110T162621_two_agent_trial/review.json`中，A未主动移动；候选128留33，Molmo选120号，实际308步/14.40秒后仍为空抓。B保存4个真实表面参考，但腔体底面/净空仍未知，未放置。另一次明确以移动取景为目标的独立诊断`runs/20261110T163742_a_observation_motion_trial/review.json`中，A自主上移18cm并自动保存4组周期双图；不过8轮内查询25次坐标仍未交稿，因此只验证了移动和历史保存，没有完成该诊断的B交接。这些现象表明，开放工具和提供角色说明尚不足以保证A高效取景，候选筛选/视觉选择也不认证真实抓握。所有原始失败与视频保留，不继续自动重跑凑成功。

首轮旧分工的真实task9记录：`runs/20261110T144024_two_agent_trial/review.json`。A用32次反投影后提交把手抓取；128候选保留9个，Molmo选择119号，实图确认空抓。B用1次API/122.14秒完成文字预分析，与物理步进窗口交叠21.26秒。该轮没有真实放置，也没有当前的主动观察采样。

中间仅符号A版本的真实记录：`runs/20261110T151011_two_agent_trial/review.json`。A只调用提交，一轮34.51秒，零感知查询。B保存14个表面坐标查询，但第8次API连接失败，未提交最终参考集合，不将零散查询当成放置点缓存已验收。Molmo选41号后，当前到预抓点路径IK失败，抓取动作0；未执行放置。B与候选准备流程交叠126.40秒，实际物理交叠0。该版没有给A运动权限，现已由上面的主动观察入口替代；其56项CPU结果不替代当前真实验收。不自动重试付费请求或放宽控制阈值。

## 当前抓取草图 → 常驻 Molmo 位姿选择试验

这是 `grasp_trial` 独立静态诊断，不是完整 task9 执行。正常高层会话的七个动作不变。诊断分两阶段：

```bash
PYTHONPATH=src .venv-grasp/bin/python -m sketchik.grasp_trial plan \
  --observation runs/<当前task9观测> --env-file .runtime/agent.env
PYTHONPATH=src .venv-grasp/bin/python -m sketchik.grasp_trial select \
  --trial runs/<grasp_trial目录> --gpu-uuid <UUID> \
  --peer-pid <SAM_PID> --peer-pid <GraspGen_PID> --peer-pid <Molmo_PID>
```

- `prompts/grasp_trial.md` 是高层真实输入说明。现在同样开放`observe/get_state/read_image/look/segment/back_project`六个只读感知工具，可多轮检查后提交`submit_grasp_sketch`；不强制调用感知。输入是保存的静态观测，`observe`不会产生新视角或移动机器人。默认上限40轮/100次工具，`plan --max-turns/--max-tool-calls`可调整。高层确定当前应抓对象、部位/方式，提交原有动作草图；位姿仍由后续GraspGen链绑定，当前查询的Mask不会隐式替换后续目标。原始API回复、完整说明、逐次工具回执/耗时、实际图片和会话历史保存在run中。
- 身份描述用简洁的类别、外观及必要方位/参照；不混入抓法推理。SAM3接收高层显式`sam_prompt`（缺省使用target_description），保留类别和必要外观/部件/方位线索；Mask选择Molmo同时接收完整`target_description`和实际`sam_prompt`，分别用于辨认所属物体及所需分割范围。高层可自行选择整物或可见部件；部件Mask仅代表局部可见几何，GraspGen实际接收该Mask点云，其余场景保留用于现有邻近筛选。这与旧项目将整段描述再拼接给SAM的规则不同，实际输入分别保存。SAM3保留两图全部有效Mask；`prompts/mask_selector.md`让Molmo选对应同一目标的候选，每相机最多一个，可拒绝唯一候选或两个都拒绝。程序不做左右排序，不强迫最高分。选择是模型判断，不能认证跨视角身份。
- 选中Mask的所有有效RGB-D点进入GraspGen；可见点不补隐藏几何。原始128候选保持完整，按已有场景/IK筛选保存每项拒绝原因。每个保留候选都产生一张独立的双视角实体夹爪照片叠加图，原始编号不重排。`--stills-only`跳过动画，没有为模型选择生成无用视频。
- `prompts/grasp_selector.md` 是独立的**抓取位姿选择说明**，不是Mask说明。常驻Molmo看原始双图、全部保留候选图和高层grasp_preference，返回原编号或null/NONE，必须给可见理由。超过14候选时全部分组展示，再比较各组选择；不暗中只给最高分。不是训练过的抓握评估器，可能选错或拒绝所有候选，真实能力需由实验验证。
- 位姿选定后，按机器人模型实测的base→grip_site转换，绑定高层符号动作的位置和完整朝向，保存`bound_numeric_plan.json`。仅保存、不派发动作；没有闭爪物理验证、完整机械臂路径碰撞或后续放置实现。0动作不能算task9成功。
- Molmo服务是`sketchik.perception_service molmo`，FP16、最多2 crops/image、最多128输出token（不是必须输出128；理由要求不超过12词）；与SAM3/GraspGen同卡常驻。加载/暖机与每次真实推理分别记录。只有显式合法JSON ID才接收，不从自然语言猜编号；允许NONE且不自动重问。
- 稠密点云在GraspGen原版预处理会建立N×N GPU距离矩阵。当前wrapper将相同的“20近邻L1平均距离<0.014m”规则换成CPU cKDTree，保留点次序，不做体素降采样；官方diffusion-discriminator保持不变。服务返回输入/保留点数和CPU过滤耗时。`/release_cache`仅释放空闲CUDA缓存，模型保持常驻。
- 密钥位于git忽略且权限0600的`.runtime/agent.env`，不进提示/日志/仓库。用户授权试验采用之前第三方`deepseek-v4.1-flash`、reasoning=`low`，仍有推理，不称no-reasoning。无自动付费重试。

本轮真实验证见 `runs/20261109T031758_grasp_trial/trial_review.json`。高层先交草图，SAM/Molmo选中双视角目标后，GraspGen生成128姿态，场景/IK保留34个，34个均渲染，Molmo按14/14/6分组后再选最终ID92。**选择仍不可靠：看图有明显夹空嫌疑，不能接受模型“包住杯身”的理由作为抓握认证。机器人动作0，尚无真实抓取/放置结果。**64token截断和长句SAM零检测等失败记录保留。41项CPU检查与真实GPU感知/渲染分开记录；步骤复用后的6.09s不是端到端耗时。


### 当前实际提示入口与操作说明补充

普通数值动作会话加载 `prompts/system.md`；静态对象级抓取试验加载 `prompts/grasp_trial.md`，本次已在该文件补齐现有接口实际可用的操作说明，直接进入高层system，不要求高层调用未开放的读文件工具。两入口的工具集合不同，不把普通会话工具写成抓取草图可调用工具。

补充了双视角/机器人状态的含义、目标与部件身份分工、依据后续任务比较把手/边沿/主体、接近与夹合净空、SAM与两次Molmo选择的区别、GraspGen→原生EEF变换、pregrasp接近轴、当前五种草图动作的实际参数/默认值/预算及提交含义。明确抓取偏好应包含选择部位对后续任务的作用，没有固定抓取部位或高度配方。Molmo位姿提示补充指垫闭合路径与可见目标材料的核对，仍允许NONE。

这次仅补现有功能的说明，不增加动作、自动放置、自动恢复或服务接口，不改筛选阈值、模型配置或执行逻辑。未启动GPU/API新实验，不能据说明更新宣称抓取效果改善。旧run内的实际system_prompt快照保留原样，更新只影响以后新请求。


### 显式执行选中的抓取

用户要求实抓时可运行 `python -m sketchik.grasp_trial execute --trial <trial> --gpu-uuid <UUID> --peer-pid <各常驻模型PID> --lift-test-m 0.05`。原数值抓取计划保持不变；可选0～0.10m世界Z抬升只作为单独标注的验收后缀，默认不加。执行前重建同一任务/seed/init_state，逐项核对机器人状态、原生双图、深度及相机标定；不匹配则不派发旧抓取位姿。保存连续视频、每步关节目标/实测状态、各动作耗时和截图。命令收敛与视觉抓取结果分别记录。

发现原step0对象尚未经过重力稳定：第一份92号计划已真实执行到位，但空抓；张爪等待后固定前视可见表面高度比step0下降约7cm。当前支持`sketchik.cli probe --settle-steps 30`，在首次观测前以固定初始关节目标、开爪推进30物理步；元数据保存此prelude，execute会在同一重置中逐步重放再核对观测。它只稳定初始化，不定位目标或访问物体GT，默认settle-steps仍为0，不静默改变旧实验。


复核上一轮抓取可用 `grasp_trial plan --observation <新观测> --env-file <私有配置> --review-trial <上一轮trial>`：向同一个模型提供原始规划回合、真实工具结果、旧候选与执行截图、新观测。模型在grasp_review评估旧候选是否符合其抓取意图，并在新的grasp_preference明确两指接触与夹合要求；不由脚本指定结论或黑名单候选。旧执行状态不冒充当前观测，官方任务成功信号不传给高层。
