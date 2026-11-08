# SketchIK

独立的机器人解析动作执行项目。目前接通 **高层会话/工具接口 → LIBERO 真实观测 → 数值动作计划 → IK → 关节轨迹 → 仿真反馈**，之后再接对象级草图的抓取与放置 Grounding。

已实现兼容 Chat Completions 的高层 Agent 会话与独立操作说明，尚未通过真实收费模型验收。当前没有 VLA、SAM3、Molmo、memory、World Model 或异步规划。不导入 AsyncSketch / HarnessVLA 的执行器、会话、memory 或纠错代码。

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

**没有自动环境避障、自碰撞检查或物体碰撞模型。** 多路点可表达绕行路线，但路线需要调用方给定；终点可达不等于途中无碰撞。这里也没有抓取位姿评分、持物认证、容器可放区域估计或自动抓放技能。

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

启动前选择实际空闲的 GPU UUID。GPU 仅用于 EGL 渲染，IK 和物理仿真主要在 CPU 上运行。

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

高层共有 **12 个工具**：7 个基础动作、`observe`、`get_state`、`back_project`、`execute_plan`、`finish`。动作底层仍是之前的七个入口；没有新增一种运动控制器。

### 每次实际传入模型什么

首次请求：本项目 system 说明、工具 schema、官方任务文字、标注相机名/图像尺寸/source_step 的前视与腕部 RGB、当前关节/末端完整姿态/手指状态/夹爪命令、工具朝向轴、剩余动作额度，以及相机标定。

后续请求：保留完整会话历史，加入模型上一轮调用、真实工具回执。每次基础动作或 execute_plan 返回后，再加入最新双视角 RGB 与机器人状态。只读 get_state/back_project 不额外渲染；observe 可以主动获取新图。**不逐仿真步请求模型，也不要求每个 primitive 都经过一次模型检查**；execute_plan 可以让多个动作在本地连续执行，出错即停止后缀。

深度图保存在本地。高层通过 `back_project(observation_id, camera, pixel_xy)` 获取该观测像素的米制世界坐标；这是可见表面点，不是自动抓取点。历史测量返回原始步数和 current 标志，不静默伪装成新观测。官方 success 不发给模型作 oracle，只留在本地评测结果中。

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
