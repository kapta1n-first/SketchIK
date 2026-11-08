# SketchIK

独立的机器人解析动作执行项目。目前接通 **高层会话/工具接口 → LIBERO 真实观测 → 数值动作计划 → IK → 关节轨迹 → 仿真反馈**，之后再接对象级草图的抓取与放置 Grounding。

已实现兼容 Chat Completions 的高层 Agent 会话与独立操作说明，尚未通过真实收费模型验收。另有 SAM3＋GraspGen 常驻服务与只读候选预览，尚未接成自动抓取动作。当前没有 VLA、Molmo、memory、World Model 或异步规划。不导入 AsyncSketch / HarnessVLA 的执行器、会话、memory 或纠错代码。

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

高层共有 **16 个工具**：7 个基础动作，`observe`、`get_state`、`back_project`、`execute_plan`、`finish`，以及 `read_text_file`、`write_text_file`、`list_dir`、`read_image`。动作底层仍是之前的七个入口；没有新增一种运动控制器。

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
| 输入配置有差别 | view_env_state 可附 policy 图、1024 前视图、1024 腕部图；前视和腕部来自两个相机，不是三个相机 | 当前默认两张 256 RGB；--size 可调，但尚未按原版高层高分辨率输入做同条件验收。不能宣称观测完全一致 |
| 尚缺工程能力 | 长会话压缩 | 当前保留完整历史；长任务可能超出模型上下文 |
| 尚缺工程能力 | 多种 Planner 后端（API、Codex、Claude Code 等） | 当前只有 Chat Completions 兼容客户端，不提供这些独立 Planner 接入 |
| 尚缺工程能力 | 环境服务/RPC、Dashboard 和取消/交互等运行管理 | 当前主要单进程同步 CLI，不具备原版整套服务与交互管理 |
| 尚缺工程能力 | 更完整的评测、经验导出与 replay 等流水线 | 当前只有单轮执行和证据记录，没有整套批量评测流水线 |
| 用户暂缓 | memory/经验管理 | 暂不接；本轮 workspace 文件不等于长期经验库 |
| 主动改变方案 | 原版解析运动＋冻结 VLA；有 SAM3 服务与分割工具 | 当前 IK＋关节控制，没有 VLA；另有 SAM3＋GraspGen 候选预览，尚未接入高层动作会话。不用 VLA 是路线选择，不是少写一个文件夹 |

对象级草图、执行时自动 Grounding、三层恢复和异步规划属于本项目的研究设计/之前 AsyncSketch 的新增机制，不能算作 HarnessVLA 原版已具备而新项目漏实现的模块。当前 execute_plan 只是数值动作计划，不能冒充对象级草图。独立的新项目尚未迁移这整套机制。

同样，稳定的自动抓取/放置位姿生成、抓取评分和完整场景避障不能未经源码/实验依据就列为 HarnessVLA 已独立解决的模块。原版借助 Agent、几何工具和 VLA 完成操作；我们不用 VLA 后，需要自行补的操作能力，应与复刻框架的工程差距分开讨论。


## 常驻分割与抓取候选预览（2026-11-09）

`src/sketchik/perception_service.py` 是两个独立的本地 HTTP 服务。模型各加载一次，后续请求复用实例，不在每次 primitive 前重新加载。SAM3 返回全部超过 0.3 阈值的候选 Mask、检测框与分数；Mask 以无损 packbits/base64 传输，可恢复为与输入 RGB 同尺寸的布尔数组。没有在服务端取最高分。当前预览多个候选时要求明确指定 ID；唯一候选只用于预览，不认证物体身份。此预览尚未接入 Molmo 或高层自动目标选择。

完整链路：保存的 RGB → SAM3 Mask → Mask 内所有有效深度像素 → 内外参反投影 → 可见物体点云 → GraspGen 的 Panda diffusion-discriminator → 128 个 4×4 位姿及模型评分。这里保留点云，不用中位数将物体压成一个点。GraspGen 内部按官方逻辑清理离群点并采样；输入不包含仿真物体 GT。模型评分不是实际成功率，也不代表可达或无碰撞。

当前两个服务在 GPU2（UUID `GPU-d3ae1b42-968e-7d78-e774-7626c1233431`）常驻：SAM3 `127.0.0.1:18761`，GraspGen `127.0.0.1:18760`。`GET /health` 查看实际 PID、加载耗时和请求计数，`POST /infer` 执行推理。PID 与启动参数保存在 `.runtime/perception/`、`.runtime/*_launch.json`；日志在 `.runtime/sam3_service.log` 和 `.runtime/graspgen_service.log`。服务独立于本次终端，但尚未配置系统开机自启或崩溃自启。重启前必须重新检查 GPU，`--peer-pid` 仅允许同项目、同用户的已驻留模型服务；不会停止别人的进程。

首次部署使用 `.venv-grasp`，通过只读 pth 复用已有第三方 torch 2.7.1+cu126 等基础包，其他兼容依赖隔离安装；没有修改旧项目环境。PointNet2 扩展以 CUDA12.3 编译。GraspGen 使用 diffusers0.11.1、huggingface-hub0.25.2、transformers4.44.2；无需 FlashAttention。官方可信 checkpoint 加载显式沿用上游 torch.load 的非 weights-only 语义。SAM3 沿用已安装第三方包和只读权重。上游源码及模型留在 `.runtime`，不提交权重或环境到 Git。

来源：[NVlabs/GraspGen](https://github.com/NVlabs/GraspGen)，源码提交 `2dd8852e1be60f5f9d277fafcc621835cdf59110`；Panda 权重与样例来自 [adithyamurali/GraspGenModels](https://huggingface.co/adithyamurali/GraspGenModels)。当前明确调用原始 diffusion-discriminator，不是上游 demo 新默认的 GraspMoE/OBB。源码没有修改，扩展编译产物除外。权重校验值、包版本与本次服务状态保存在 `.runtime/graspgen_setup.json`。

生成可旋转的三维预览与 PNG：

```bash
USE_FLAX=0 USE_TF=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv-grasp/bin/python -m sketchik.grasp_preview \
  --observation-dir runs/20261108T180605_smoke --camera agentview \
  --target 'a cylindrical can with a red and green label'
```

也支持 `--sample-json` 读取官方真实点云样例，仅读取 pc/颜色，再实时推理，绝不拿样例存储的抓取结果冒充模型输出。每次新建运行目录，保存原图、完整 SAM 回应、Mask 叠图、选中 Mask、标定点云、原始候选/评分、三视角线框图、前三个 Panda 实体夹爪渲染、交互式 HTML 和耗时。`--candidate-id` 可显式选择多个候选中的一个。此程序不推进机器人，不调用 IK，不执行抓取。

本次真实 GPU 结果：

| 输入与阶段 | 结果 | 推理墙钟 |
| --- | --- | --- |
| 官方真实点云样例，GraspGen 首次推理 | 128 个候选 | 5.408s |
| 保存的 LIBERO task0 前视 RGB，SAM3 首次成功推理 | 1 个 Mask，777 个有效三维点 | 0.898s |
| 该 LIBERO 点云，GraspGen 已暖机 | 128 个候选 | 0.161s |
| 同一保存输入重复三次，SAM3 | 同一 PID，均返回 1 个候选 | 0.221 / 0.198 / 0.187s |
| 同一点云重复三次，GraspGen | 同一 PID，均返回 128 个候选 | 0.172 / 0.143 / 0.144s |

这些是当前 256 RGB、单物体、128 抓取候选的推理耗时，不含模型加载和绘图，不能推广为完整机器人任务耗时。两服务加载后 GPU 总占用约 6.6GiB（含缓存和 CUDA 上下文）；torch 单进程峰值 allocated 为 SAM3 5088MiB、GraspGen 737MiB，不能把它与 nvidia-smi 总占用混为同一指标。

LIBERO 证据：`runs/20261109T015914_grasp_preview`，含 `metadata.json`、`result.json` 和 `resident_repeats.json`。官方样例证据：`runs/20261109T015805_grasp_preview`。首次 SAM 请求曾因 BF16 转 NumPy 报错，已修为 float32 后序列化并重启本项目服务，错误日志保留。最终 29 项 CPU 测试通过，包含所有有效 Mask 像素的三维转换和多候选不默选最高分。

**本轮只验证生成与显示。** 部分高分罐子候选从下方接近，可能与桌面碰撞；没有场景碰撞筛选，不能直接执行。GraspGen 输出是其夹爪基座约定，后续必须对齐本机器人 grip_site/TCP，检查 IK、关节与场景碰撞，再验证真实抓取。渲染图不是已成功抓住的画面。
