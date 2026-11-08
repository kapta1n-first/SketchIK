# SketchIK

独立的机器人解析动作执行底座。第一版先把 **LIBERO 观测 → IK → 关节轨迹 → 真实仿真执行** 接通，之后再接符号草图、抓取与放置 Grounding。

当前没有 Strong Agent、VLA、SAM3、Molmo、World Model 或异步规划。也不导入 AsyncSketch / HarnessVLA 的执行器、会话、memory 或纠错代码。

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

`examples/basic_motion.json` 是可直接运行的冻结基础动作示例，**不是抓放任务方案**。JSON 使用 `version=1` 和 `steps`；每步有 `primitive` 与 `params`。执行器不接入付费 API。

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
