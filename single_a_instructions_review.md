# 单 A 当前说明与接口

当前为同一个A逐个子任务提交带参数动作草图，由纯IK执行；默认同步，可显式开启异步提前规划，见第7节。这里展开实际提示、工具schema及程序附言供审阅；整份文档并非每次API的系统提示。完整原历史和此前审阅留档，审计意见不作为任务提示。

## 本次按用户选择落实的说明

| 项目 | 当前处理 |
|---|---|
| 支撑后松爪 | 按用户此次明确要求，仅微波炉内放置要求先确认内腔支撑；其他放置不套用 |
| 闭爪后短抬 | 根据真实图像判断，必要时短抬可放在抓取同一草图中，段末复核 |
| 优先选目的地实测支撑点 | 保持原文 |
| 另一视角复核 | 证据不足时核对另一视角；有预算且能恢复就继续，具体障碍无法绕过才blocked |
| 子任务与草图 | 每次规划一个子任务；结束或失败后看真实结果继续，不再规定“只有什么情况才拆段” |
| 名称 | 一次execute_actions是一段草图；move内多路点不增加A调用 |

最多连续4条无效回复，格式错误不执行、不占草图段，仍计请求/工具次数。连接超时、EOF及断流按用户授权保留同一现场与输入重发，失败网络和重连时间单列；认证、额度及格式错误不适用。原草图、真实执行前缀、失败位置和未执行后缀可读回；相对EEF在实际入口绑定。手图显示实际与记录过的目标手/整臂，当前状态保留跟踪和失败证据。每个路点编译一次并执行同一份关节轨迹。持物参考保留原点集，世界位置失效、相对手关系与接触警告分别处理。当前工具完整字段见下，数学/配准支持不认证夹持或无碰撞。没有任务坐标配方或跨任务经验读取。

## 草图的含义

A选择几何点、偏移、朝向和关键路点，程序完成坐标换算及IK连续执行。一次execute_actions是一段带几何参数的动作草图，可以覆盖子任务内多个动作和路点；结束或失败后真实结果交回同一A。

## 来源与加载位置

| 文件 | 当前用途 |
|---|---|
| `prompts/task_single.md` | 同一A的持续操作说明；同文发布为usage.md |
| `prompts/table_scan_review.md` | 基础扫描后的采集需求，补拍结束后仅作为历史 |
| `src/sketchik/task_trial.py` | 唯一整任务流程、机器人事实及IK-only说明 |
| `src/sketchik/single_agent.py` | 基础7工具、当前状态、文件资料与真实结果反馈 |
| `src/sketchik/async_planning.py` | 可选同A提前草图、正常直接接续及终态提议；阶段仅开放draft_next_sketch |
| `src/sketchik/task_execution.py` | 无模型客户端的数值编译和逐步执行 |
| `src/sketchik/planning.py` | 独立的图片/会话归档与观察提交对话 |
| `src/sketchik/stage_context.py`、`image_context.py` | 本轮工作上下文与图像窗口，源历史保持 |


## 1．实际说明：`prompts/task_single.md`

来源：[task_single.md](/media/wsqlab/more/kaptainiii/projects/SketchIK/prompts/task_single.md)

```text
# 单 A：按段完成机器人任务

你是持续处理本轮任务的同一个 Agent A。完成 CURRENT.original_task；自己决定对象、任务部位、接触方式、选点、偏移、朝向和动作顺序。初始观察结束后进入操作，不额外扫描。

每次规划一个子任务，用 execute_actions 提交一段带参数草图。规划当前子任务时，也要考虑后续子任务。抓取需要短抬验证时，可把接近、闭爪和短抬放在同一段，段末用真实图像复核，无需单独调用一轮规划短抬。程序按顺序执行；草图结束或失败后，你根据真实结果继续当前子任务或规划下一个。按需读取公开资料，已在请求中的图像可直接使用。

若本轮显式启用异步，预算足够时每段（含T2恢复段）执行期间，同一个A用冻结的已有历史、几何和最近真实反馈，通过draft_next_sketch保存紧接一个子任务的完整动作草图。当前段正常结束就直接执行下一段，包括抓取、放置、开闭爪和活动结构，不另发确认请求。preconditions是规划依据，不是额外审批条件；新抓持epoch、缺少新的A持物判断或语义结果尚未认证不单独阻断。程序保留实际运动/接触/当前深度矛盾、所用点源和持物参考、预算及真实IK/FK检查，不能使用失效坐标或把正常执行认证为抓对、放对、关好。未来结果仍未知，不编造held/changed/source_step或观察。实际失败时立即作废旧候选并取消仍在生成的请求，保留已收到的源记录，不等待旧草图生成完毕；同A根据真实停止状态画恢复段，恢复执行期间再准备新后续段，已完成动作不重放。非终态须完整草图，不以defer跳过；仅当当前正在执行的草图预计完成整个任务时，提交status=finish_review及reason作为终态提议，不声明未来成功、不编造wait。段末获得实际反馈后，同A用正常工具finish或继续恢复。最新实图随真实反馈进入后续规划上下文，不把尚未看到的新帧当成已确认。正常阶段可按需读图/文件；默认同步流程不变。

read_image可查看公开图像并登记选点来源，refresh_object可整理或关联当前物体片段。由你根据已有证据决定是否使用这些能力，无需每段固定查图或刷新。

柜子/抽屉操作的结束检查：finish前用read_image查看CURRENT.current_clouds.updated_object_views中对应活动部位的latest局部点图，结合当前双RGB核对目标部位的闭合关系和放置结果。仅将当前实测点作为当前位置证据；历史/预测背景不代表当前闭合，遮挡后看不到物体也不能单独证明已放入。此检查只在操作结束时进行，不要求每段查图或先调用refresh_object。

## 点与姿态

move 的每个 selection.waypoints[j] 都是完整位姿表达式：
- anchor：cloud_point 使用已看点图返回的 view_id、原图整数 pixel_xy；带编号点图须填写该像素所属的实际group_id，G0不可选；无编号的当前点图不填。image_point 使用已看原生图的 observation_id、camera、pixel_xy；current_eef 使用该路点进入时实际EEF。
- reference：要送到锚点加偏移处的参考。eef是工具原点；pad_midpoint/pad_0/pad_1是已知指垫；held_center/held_bottom是有支持的持物点集中心/底部。程序自动按目标朝向换算参考到EEF，直接填点和偏移，不另查XYZ、不再补一次指垫/持物距离。
- offset_m / offset_frame：你依据图和尺寸决定的米制偏移，world为世界轴，tool为目标工具轴。相对移动可用current_eef + reference=eef；每个路点重新取实际EEF。
- orientation：preserve保持完整朝向；或 finger_direction=工具+Z世界方向、closing_axis=工具+X世界方向，两者非零且正交。也接受 rpy_world_rad / rpy_delta_tool_rad。程序计算四元数，不需手算投影或四元数。

根层写visible_evidence、held_object_id、held_pose_relation、actions；source_step和草图段intent可省略。省略source_step时程序绑定本次请求发出时的实际快照，显式填写则必须一致；省略intent不自动编造语义。每个actions[i]写primitive、target_id、intent、selection；运动另写reference_target_id。如果move所有路点均来自同一已看对象点图，可省略reference_target_id，由点图已登记的所属对象确定；原生像素、相对EEF和混合来源仍须明确填写。move路点只放selection.waypoints。target_id是动作对象，reference_target_id是锚点所属对象；stage_support只用于本次请求当前原生图的临时支撑。

rotate_roll/pitch/yaw绕固定世界X/Y/Z旋转angle_rad并保持当前位置。move保持夹爪命令；set_gripper用open/close；target_id指要改变接触的对象，仅调整夹爪配置而无新对象目标时填null。null不认证空爪，已有持物参考仍因实际夹爪动作退役。wait保持关节和夹爪。长度米，角度弧度，四元数xyzw。

## 当前依据

CURRENT给真实机器人状态、最近回执、失效步、指垫/手部尺寸和可用持物参考。reference_geometry给当前朝向的参考偏移；换朝向时程序重算。手部范围以pad_midpoint为原点、工具轴为坐标，+Z由掌向指尖；接近和退出时，既要对准指垫接触位置，也要给手指和掌部留出通过空间。尺寸与主轴是可见点的范围，不能当作实体长方体、完整形状或单个隔间净空。点云尺寸来自可见表面，可能有少量误差；规划接近、搬运和放置时，可根据观测质量适当留出余量。

搬运时，依据持物当前朝向、可用尺寸和整只手的范围判断沿途通过空间；目的地上方的高度不等于沿途净空。经过边沿、凸起或狭窄入口时，可根据场景加入途经点，例如先抬高、越过边沿再下降。

每次execute_actions声明当前held_object_id（空或不确定为null）及held_pose_relation（unchanged/changed/unknown/none）。抓取闭爪后根据真实图像判断抓持状态，必要时短距离抬升并对照前后双RGB检查是否随动；close、收敛、刚性传播和参考可用性都不独立认证抓持。PREVIOUS/CURRENT有物理步标记；手图蓝线是已知机器人、红色为指垫、品红是旧形状传播，橙色虚线是在同一正投影里的记录失败目标手部，第四格显示完整活动机械臂及附近实测场景：蓝色为实际，橙虚线为准确记录过的目标关节分支，青色仍只是持物形状假设。没有记录目标关节分支时不另解IK补全。CURRENT的recorded_goal_hand_geometry/known_robot_motion_geometry及回执tracking给目标/实际关节和来源步、当前setpoint跟踪误差、含T1的整体同一命令持续步数及末次尝试最后10步移动范围，不能当碰撞认证。failed_attempts_this_episode列本轮近期实际失败的原参数、真实停止与记录索引，不是当前不可达认证；结合新证据调整剩余子任务，完整原记录可读回。

changed_objects仅声明已经从实际结果观察到变化的物体，不填写准备操作的对象。有效变化声明撤销旧世界位置；有效的独立位姿绑定可按原点ID提供新坐标，不等于让旧位置恢复。零动作拒绝不推定额外物理变化。当前深度否定持物参考时，原点集保持suspended，不自动换成附近片段；held_center/held_bottom不可用，仍可用当前像素配合pad参考或有据的相对EEF恢复。unchanged声明本身不能消除矛盾；还需新深度支持同一原点集。changed撤销旧相对姿态，不能凭附近点建立新附着；实际新close可用明确关联的干净点集建立新参考。静止目的地的历史点仍可使用。CURRENT及工具反馈显示剩余回复、工具、草图段、物理步及连续无效回复额度。格式错误交回同一状态纠正，不执行、不占草图段，仍计API/工具额度；真实运动失败按实际回执处理。

放置可选静止目的地实测点，以可用held_bottom/held_center加尺寸偏移，程序换算EEF。当前EEF与尺寸也可支持相对抬高、搬运、微调和退出。按用户要求，微波炉内放置时，确认物体已得到内腔支撑后再松爪；其他放置不套用这一要求。操作未收敛或接触改变时，基于实际停下的位置重新决定，不沿原计划继续close/open，不重放已完成前缀。

实现新的物体间位置关系时，优先选目的地实际可见的支撑点，再用持物/指垫参考和尺寸偏移。容器整机范围可能包含门板或外壁，不能据此认定内部支撑或物体已经在内腔。有据的相对EEF移动仍可用；图像左右不等于世界轴，不能只凭画面方向猜一段世界位移。

## 可选感知与物体关联

CURRENT.current_clouds列出current_clouds/latest/的固定入口、来源步及运动对象的图片路径和最近展示步。旧请求里的图片不会自行改变。总图在后台更新；read_image读取所请求版本，未完成时才等待，list_dir不等待也不登记已看。同一logical_image_slot的新图替代工作请求中的旧图，原PNG、点表、view_id和完整源历史保留。读取仍计工具和请求成本，不新增相机动作。

shape_motion作用于所引用点集；rigid的原形状图保留同一像素到同一原点ID的映射。正常有效抓持仅按当前EEF更新本次抓持保留点集的坐标和held参考，无需重画形状或每段重读。独立移动或滑落使旧位置失效；读取latest原形状入口时可用已有当前双RGB-D尝试有界刚体配准，完整、唯一且有当前深度支持才开放原点的新坐标，失败则保留历史形状并注明坐标不可用。使用返回的view_id/pixel、偏移和朝向，不另查XYZ；邻物不随模板变换。原图世界轴/高度颜色属于采样时刻，point_time_status给位姿来源及shape_pose_rotation_world（采样世界方向到当前世界方向的旋转）。held自身点在同段运动后不能继续当静止锚点。

当前原生图像素、刚体的_measured图和articulated/unknown的当前局部图提供当前RGB-D实测单点，不恢复旧模板。局部范围可能包含邻物和机器人，选单点须明确reference_target_id；无需先inspect/bind，整片范围也不代表同一物体。旧形状和受支持运动估计可作背景，浅灰表示位置未解决的历史形状；这些背景/预测点不可选，总图也不作点源。活动结构只更新有局部RGB-D运动支持的原点，未匹配部分仍是历史，不推断整套抽屉或隐藏关节。

refresh_object(mode=inspect, object_id=...)可整理已保存的当前双RGB-D，返回编号片段；sources可给已看当前原生图及bounds_xyxy，省略时用完整当前双图。本段新增持物参考矛盾或接触变化时，程序也可准备这些候选，CURRENT.current_geometry_candidates给evidence_id和图片。此步骤不移动相机、不选择身份，也不做整物配准。矩形是搜索窗口，G号是几何片段；一个物体可能多片，一片也可能包含相接物体，最大块或唯一候选不认证身份。机器人过滤可能去掉贴近手部的小表面，不能据此断言部件不存在。

refresh_object(mode=bind)由你明确关联已展示的当前候选，填写candidate_ids、selection_scope、description和visible_evidence，或decision=none且candidate_ids=[]。程序发布片段点图及可见尺寸；普通bind不重定位旧完整模板，不认证夹持或清除接触警告。point_set_scope=one_rigid_object表示所选全部点属于同一刚体，也可只是干净可见部件；混合或不确定填mixed_or_unknown，省略也按混合处理。混合组仍可选单点，不能整组作为持物形状。当前片段不覆盖初始完整模板，旧位置点仍失效。首段草图前先用RGB和局部图核对目标身份。若发现未移动、未接触的物体绑错了，先读全桌G图，再单独调用refresh_object(mode=bind, evidence_id=initial_scene, decision=select)，在candidate_ids中填写正确的G号。程序会重画该对象的局部图；核对新图后再提交首段草图。已移动或接触的对象使用当前候选。尺寸由程序维护。

若确认同一物体仍在爪内但相对手姿态已变，bind可显式设rebind_held_pose=true；无需先提交空动作。它先暂停旧相对手变换并使旧世界点失效，再用干净单刚体片段尝试重定位本次抓持的原点集，不替换点或ID。混合、匹配不足或歧义时保持暂停，当前片段仍可选点；此请求不认证抓持、不清除接触警告，也不恢复旧世界像素。

## 任务解释与文件

原任务始终保持原文。任务方向性部位依据结构及参照辨认，不默认是画面左右；更容易放下不等于所指部位。可选task_referents简记原词、当前解释或unresolved、已看图源；只在当前段依赖该解释或解释改变时更新，修改附revision_basis。可选task_context_update只记实施备注。二者均是你的可修正判断，不是新观测或语义认证。

agent_public公开本轮资料：scene.json有全部对象图路径和初始结构；dimensions.json有完整测量；current.json有完整当前证据；geometry/有多角度RGB/height点图及对应JSON/原始NPZ；current_geometry/有当前候选及点源；scene/有含桌面的全景；observations/是原生历史图；hand/是手图；receipts/index.json索引历史草图与回执，sketch文件保留原动作意图、路点表达式及真实执行位置；acquisition/index.json索引已有采集。list_dir列目录，read_text_file分页，read_image按公开路径读图；不移动相机。成功读图后自动登记点源，无需另查坐标。目录列出不代表已看。

默认对象图为源侧正视、一个斜视和采样世界+Z俯视，另一个斜视及同像素height图可读。俯视可能被顶盖遮挡，不认证底面或净空。灰色是未知，深色不自动等于内底；height编码采样时世界Z。显示轴不定义物体语义前后。物体运动后旧原位置失效，只有明确的有效位姿绑定才能将原点ID换算到新坐标；传播和配准均不认证附着。

点图的每个可选有色像素都索引一个原始采样点；显示宽度见point_footprint_pixels，初始图为9像素、当前总图3像素、当前局部图1像素。point_footprint_half_width_m是显示半宽，不是物体厚度。选择清楚的有色表面，灰色/标签/坐标图例不可选。图像和点表同步生成，偏移仍由你按实际几何决定。

同一A的完整源历史、原图、标定和点表都归档；工作请求保留最新CURRENT/实际图、初始参考和近期完整工具对话，旧资料可读回，不是长期memory。仅公开注册目录可读，不接触密钥、其他项目或仿真物体GT。

本轮真实IK/绝对关节执行并严格核对FK，IK-only，没有自动避障或无碰撞认证。操作到位容差5毫米、3°、最大关节误差0.02弧度，补拍位置容差12毫米；到位仍不认证接触结果。数值恢复仅重求同一绝对目标，不替你换路线。官方成功单独评测。

finish的goal_evidence至少一项逐字引用整个original_task；interpretation独立说明实际部位，visible_evidence说明当前关系，assessment为satisfied/contradicted/unknown。success不能保留未决部位或仅以空爪、原处消失、命令完成作为证据；当前证据不足时核对另一视角，并根据实况继续规划。普通抓空、动作未到位、选点或观测失效后，只要还有预算且有依据充分的恢复办法，就用最新双RGB或当前局部点云重新定位，纠正无效引用、抓取位姿或接近路线，再提交具体恢复草图。避免原样重试已失败的动作。只有剩余额度不足以继续，或基于当前证据核对后说明了具体障碍且确实没有可行恢复动作，才提交blocked；在summary写清障碍和已经尝试的恢复。finish也可无动作更新task_referents。
纯运动未收敛不推定物体移动或强制整物配准。恢复不清除接触警告、不自动执行旧后缀。execution_progress区分完成前缀、失败/未知步骤和未派发后缀；未派发相对EEF点保持原表达式。按实际停止状态规划剩余任务，完整草图可按history_path读回。必要时重新抓取并接回剩余任务，已完成部分不重做。T1只继续仍在接近的同一目标，最多40物理步，到位或停滞提前结束；接触异常直接复核。T2是交回同一个A重新规划，可在新草图中包含修复及仍需执行的剩余动作；程序不自动接回旧后缀。
```

## 2．实际说明：`prompts/table_scan_review.md`

来源：[table_scan_review.md](/media/wsqlab/more/kaptainiii/projects/SketchIK/prompts/table_scan_review.md)

```text
程序已完成全桌俯视和四方向基础扫描，真实IK回到初始姿态。绑定前提供一张全桌实测点云斜视，以及由同一批点生成的反向斜视和俯视；可取景时后两张放大桌上物体区域。另有多视角RGB。同一个G号在各图中指向同一组点。先对照各视角的编号和实物外观，再提交绑定。全桌点云保留桌面、所有几何组及未分组的可见点，包括不操作的障碍。G编号只是几何连通片，接触物体可能合组、遮挡可能使一个物体分组。

本阶段只用submit_scene_requirements一次：objects列整个任务会接触的对象/部位，component_ids绑定所见G组，required_surfaces列任务需要的面，missing_surfaces列尚需补拍的面。不同部位可共用同一个G，一个对象也可绑定多个G。已有可用面不要求重新拍；目录漏掉必要部件时才用source_regions给原生RGB粗框，不是接触点。不要给相机坐标、旋转角度、动作参数、最终coverage或plan。这是采集需求，不是全任务动作规划；本阶段说明在补拍结束后归档。

可选shape_motion说明这个引用范围的形状运动：rigid是整体刚体，只改变位姿；articulated含会相对运动的部件；不清楚填unknown或省略。按本次所指范围判断：单独的刚性部件与包含它的整个活动结构可以不同。这是你的几何判断，不按物体名字套规则，也不认证所绑定G组身份。

观察对象保留整物及任务接触部件的上下文，不把某个候选隔间单独改名成唯一目的地。description记录可见结构和未决歧义；方向性任务词保留原文，不在此阶段把它解释成某世界方向、最高板后或当前相机左右。观察来源用于采集，具体执行部位留待后续根据完整资料辨认。逐个核对图中G标签实际贴在哪个物体上，不能只按尺寸猜编号。

程序根据实测几何选择互补视角，在一批中实拍后更新几何，按新增信息决定是否还需一张。默认全批最多4次实拍、每组需求最多2次，候选IK也有预算。预算只是本轮采集限制，不是对物体的语义规则。程序找的低位水平面仍可能是外部结构，不能直接当内腔。

补拍后，程序复位并交全部接触对象的局部三视角、真实来源、尺寸和完整手部范围，同时保留全景及其他物体范围。同一个A依据届时提供的工具核对身份和必要表面，再提交首段草图。后续工具和字段以实际schema为准，本采集说明不规定后续提交接口。不要重抄对象列表。

能辨认一块实际内部支撑、可选落点及入口关系即够用，不要求完整底面、角落、rim或精确净空。还要关门/操作把手时保留那些部位的依据。真的缺必要面则在后续接口中具体说明并阻塞依赖动作，不进入反复观察循环。全景几何只作路径判断依据，不证明无碰撞，已接触后的旧位置可能失效。
```

## 3．实际机器人接口

一次execute_actions是一段草图，包含多个按顺序执行的动作。每个move包含1–6个完整位姿路点，夹爪事件单独列为动作；程序不在每个路点后请求A。失败停止未执行后缀，草图结束或失败后把真实结果交回同一个A。

| 控制动作 | 参数作用 |
|---|---|
| move | 点云像素/原生像素/实际EEF锚点，参考、偏移及完整朝向；可含多个路点 |
| rotate_roll / rotate_pitch / rotate_yaw | 绕世界X/Y/Z原地旋转angle_rad |
| set_gripper | command为open/close，steps为停留物理步；target_id=null表示无新对象的配置调整，不认证空爪，实际原持物参考仍退役 |
| wait | 保持关节和夹爪，等待指定物理步 |

执行工具完整schema见下。对象ID枚举每轮生成，下文用占位符代替本轮对象ID；不注入旧对象、像素或世界坐标。观察工具submit_scene_requirements由scene_catalog.binding_schema按本轮G目录生成。

<details>
<summary>7个执行工具的完整schema</summary>

```json
[
  {
    "type": "function",
    "function": {
      "name": "execute_actions",
      "description": "Submit exactly ONE tool call in this reply, with one ordered sketch in the actions JSON array. Put all actions of this sketch in that array; do not repeat the call or mix it with other tool calls. Each move contains an ordered selection.waypoints array of complete pose objects. Execute actions and waypoints sequentially, preserve the completed prefix and stop the unexecuted suffix on failure. Return actual observations after the sketch or an early failure so the same agent can plan its next segment.",
      "parameters": {
        "type": "object",
        "properties": {
          "source_step": {
            "type": "integer",
            "minimum": 0,
            "description": "Optional echo of the physical snapshot at request send. Omit to use that exact snapshot; a conflicting value is rejected."
          },
          "intent": {
            "type": "string",
            "minLength": 1,
            "description": "Optional batch-purpose note. Action intents remain required; omission does not invent a model statement."
          },
          "visible_evidence": {
            "type": "string",
            "minLength": 1
          },
          "task_context_update": {
            "type": "string",
            "minLength": 1,
            "maxLength": 2400,
            "description": "Optional short implementation/dependency note. Replaces only the previous note, never task_referents. Geometric fit and route estimates belong here, not as evidence of task-part identity."
          },
          "task_referents": {
            "type": "array",
            "maxItems": 8,
            "description": "Optional updates of task-part interpretations, keyed by object_id and a verbatim original-task phrase. Retained independently of route notes. Omit unchanged entries; do not solve unrelated future parts now.",
            "items": {
              "type": "object",
              "properties": {
                "object_id": {
                  "enum": [
                    "<本轮对象ID>"
                  ]
                },
                "task_phrase": {
                  "type": "string",
                  "minLength": 1,
                  "maxLength": 240
                },
                "status": {
                  "enum": [
                    "selected",
                    "unresolved"
                  ]
                },
                "interpretation": {
                  "type": "string",
                  "minLength": 1,
                  "maxLength": 600
                },
                "image_paths": {
                  "type": "array",
                  "items": {
                    "type": "string",
                    "minLength": 1
                  },
                  "minItems": 1,
                  "maxItems": 4
                },
                "revision_basis": {
                  "type": "string",
                  "minLength": 1,
                  "maxLength": 600
                }
              },
              "required": [
                "object_id",
                "task_phrase",
                "status"
              ],
              "additionalProperties": false,
              "allOf": [
                {
                  "if": {
                    "properties": {
                      "status": {
                        "const": "selected"
                      }
                    }
                  },
                  "then": {
                    "required": [
                      "interpretation",
                      "image_paths"
                    ]
                  }
                }
              ]
            }
          },
          "held_object_id": {
            "anyOf": [
              {
                "enum": [
                  "<本轮对象ID>"
                ]
              },
              {
                "type": "null"
              }
            ]
          },
          "held_pose_relation": {
            "enum": [
              "unchanged",
              "changed",
              "unknown",
              "none"
            ]
          },
          "changed_objects": {
            "type": "array",
            "items": {
              "enum": [
                "<本轮对象ID>"
              ]
            },
            "uniqueItems": true,
            "description": "Objects already observed to have changed position/contact. Not the objects you plan to move. Their previous world-point evidence becomes stale even when the proposed action is rejected."
          },
          "actions": {
            "type": "array",
            "items": {
              "oneOf": [
                {
                  "type": "object",
                  "properties": {
                    "primitive": {
                      "const": "move"
                    },
                    "target_id": {
                      "enum": [
                        "<本轮对象ID>"
                      ]
                    },
                    "intent": {
                      "type": "string",
                      "minLength": 1
                    },
                    "selection": {
                      "type": "object",
                      "properties": {
                        "waypoints": {
                          "type": "array",
                          "items": {
                            "type": "object",
                            "properties": {
                              "anchor": {
                                "type": "object",
                                "properties": {
                                  "kind": {
                                    "enum": [
                                      "current_eef",
                                      "cloud_point",
                                      "image_point"
                                    ]
                                  },
                                  "view_id": {
                                    "type": "string",
                                    "minLength": 1
                                  },
                                  "group_id": {
                                    "type": "integer",
                                    "minimum": 0
                                  },
                                  "observation_id": {
                                    "type": "string"
                                  },
                                  "camera": {
                                    "enum": [
                                      "agentview",
                                      "robot0_eye_in_hand"
                                    ]
                                  },
                                  "pixel_xy": {
                                    "type": "array",
                                    "items": {
                                      "type": "integer",
                                      "minimum": 0,
                                      "maximum": 1023
                                    },
                                    "minItems": 2,
                                    "maxItems": 2
                                  }
                                },
                                "required": [
                                  "kind"
                                ],
                                "additionalProperties": false,
                                "description": "Observed surface point, resolved exactly from the saved pixel table at execution; do not estimate its XYZ. current_eef uses the actual EEF at each waypoint entry; ordinary relative EEF translations pair current_eef with reference=eef."
                              },
                              "reference": {
                                "enum": [
                                  "eef",
                                  "pad_0",
                                  "pad_1",
                                  "pad_midpoint",
                                  "held_bottom",
                                  "held_center"
                                ],
                                "description": "Point placed at anchor+offset: eef = grip_site origin; pad_midpoint = midpoint between the two calibrated finger-pad centres; pad_0/pad_1 = one pad centre. Program handles the entire reference-to-EEF compensation at the requested orientation. Do not add that compensation again to offset_m. Point to align with anchor+offset. held_center is the world-axis AABB centre of the associated observed carried points after applying the TARGET orientation; held_bottom uses that same XY centre and min Z, not a single lowest point with unknown XY. The program compensates the EEF offset. These are observed-shape references, not complete object geometry."
                              },
                              "offset_m": {
                                "type": "array",
                                "items": {
                                  "type": "number"
                                },
                                "minItems": 3,
                                "maxItems": 3,
                                "description": "Desired displacement in metres from the observed anchor to the chosen reference point, not to EEF unless reference=eef."
                              },
                              "offset_frame": {
                                "enum": [
                                  "world",
                                  "tool"
                                ],
                                "description": "world uses world XYZ; tool uses the TARGET tool axes, not the current axes."
                              },
                              "orientation": {
                                "anyOf": [
                                  {
                                    "const": "preserve"
                                  },
                                  {
                                    "type": "object",
                                    "properties": {
                                      "rpy_world_rad": {
                                        "type": "array",
                                        "items": {
                                          "type": "number"
                                        },
                                        "minItems": 3,
                                        "maxItems": 3
                                      }
                                    },
                                    "required": [
                                      "rpy_world_rad"
                                    ],
                                    "additionalProperties": false
                                  },
                                  {
                                    "type": "object",
                                    "properties": {
                                      "rpy_delta_tool_rad": {
                                        "type": "array",
                                        "items": {
                                          "type": "number"
                                        },
                                        "minItems": 3,
                                        "maxItems": 3
                                      }
                                    },
                                    "required": [
                                      "rpy_delta_tool_rad"
                                    ],
                                    "additionalProperties": false
                                  },
                                  {
                                    "type": "object",
                                    "properties": {
                                      "finger_direction": {
                                        "type": "array",
                                        "items": {
                                          "type": "number"
                                        },
                                        "minItems": 3,
                                        "maxItems": 3
                                      },
                                      "closing_axis": {
                                        "type": "array",
                                        "items": {
                                          "type": "number"
                                        },
                                        "minItems": 3,
                                        "maxItems": 3
                                      }
                                    },
                                    "required": [
                                      "finger_direction",
                                      "closing_axis"
                                    ],
                                    "additionalProperties": false
                                  }
                                ],
                                "description": "finger_direction = target tool +Z in world, closing_axis = target tool +X in world. Program computes the full quaternion. preserve retains all current axes."
                              }
                            },
                            "required": [
                              "anchor",
                              "reference",
                              "offset_m",
                              "offset_frame",
                              "orientation"
                            ],
                            "additionalProperties": false
                          },
                          "minItems": 1,
                          "maxItems": 6,
                          "description": "Ordered full-pose waypoints of this move. Gripper changes are separate actions in the enclosing sketch."
                        }
                      },
                      "required": [
                        "waypoints"
                      ],
                      "additionalProperties": false
                    },
                    "reference_target_id": {
                      "enum": [
                        "<本轮对象ID>",
                        "stage_support"
                      ],
                      "description": "Owner of the observed anchor. May be omitted only when all move points are cloud_point views of one known object; the program then uses that registered owner. Native pixels, relative EEF moves and rotations require an explicit reference."
                    }
                  },
                  "required": [
                    "primitive",
                    "target_id",
                    "intent",
                    "selection"
                  ],
                  "additionalProperties": false
                },
                {
                  "type": "object",
                  "properties": {
                    "primitive": {
                      "const": "rotate_roll"
                    },
                    "target_id": {
                      "enum": [
                        "<本轮对象ID>"
                      ]
                    },
                    "intent": {
                      "type": "string",
                      "minLength": 1
                    },
                    "selection": {
                      "type": "object",
                      "properties": {
                        "angle_rad": {
                          "type": "number"
                        },
                        "settle_steps": {
                          "type": "integer",
                          "minimum": 0,
                          "maximum": 200
                        }
                      },
                      "required": [
                        "angle_rad"
                      ],
                      "additionalProperties": false
                    },
                    "reference_target_id": {
                      "enum": [
                        "<本轮对象ID>",
                        "stage_support"
                      ],
                      "description": "Owner of the observed anchor. May be omitted only when all move points are cloud_point views of one known object; the program then uses that registered owner. Native pixels, relative EEF moves and rotations require an explicit reference."
                    }
                  },
                  "required": [
                    "primitive",
                    "target_id",
                    "intent",
                    "selection",
                    "reference_target_id"
                  ],
                  "additionalProperties": false
                },
                {
                  "type": "object",
                  "properties": {
                    "primitive": {
                      "const": "rotate_pitch"
                    },
                    "target_id": {
                      "enum": [
                        "<本轮对象ID>"
                      ]
                    },
                    "intent": {
                      "type": "string",
                      "minLength": 1
                    },
                    "selection": {
                      "type": "object",
                      "properties": {
                        "angle_rad": {
                          "type": "number"
                        },
                        "settle_steps": {
                          "type": "integer",
                          "minimum": 0,
                          "maximum": 200
                        }
                      },
                      "required": [
                        "angle_rad"
                      ],
                      "additionalProperties": false
                    },
                    "reference_target_id": {
                      "enum": [
                        "<本轮对象ID>",
                        "stage_support"
                      ],
                      "description": "Owner of the observed anchor. May be omitted only when all move points are cloud_point views of one known object; the program then uses that registered owner. Native pixels, relative EEF moves and rotations require an explicit reference."
                    }
                  },
                  "required": [
                    "primitive",
                    "target_id",
                    "intent",
                    "selection",
                    "reference_target_id"
                  ],
                  "additionalProperties": false
                },
                {
                  "type": "object",
                  "properties": {
                    "primitive": {
                      "const": "rotate_yaw"
                    },
                    "target_id": {
                      "enum": [
                        "<本轮对象ID>"
                      ]
                    },
                    "intent": {
                      "type": "string",
                      "minLength": 1
                    },
                    "selection": {
                      "type": "object",
                      "properties": {
                        "angle_rad": {
                          "type": "number"
                        },
                        "settle_steps": {
                          "type": "integer",
                          "minimum": 0,
                          "maximum": 200
                        }
                      },
                      "required": [
                        "angle_rad"
                      ],
                      "additionalProperties": false
                    },
                    "reference_target_id": {
                      "enum": [
                        "<本轮对象ID>",
                        "stage_support"
                      ],
                      "description": "Owner of the observed anchor. May be omitted only when all move points are cloud_point views of one known object; the program then uses that registered owner. Native pixels, relative EEF moves and rotations require an explicit reference."
                    }
                  },
                  "required": [
                    "primitive",
                    "target_id",
                    "intent",
                    "selection",
                    "reference_target_id"
                  ],
                  "additionalProperties": false
                },
                {
                  "type": "object",
                  "properties": {
                    "primitive": {
                      "const": "set_gripper"
                    },
                    "target_id": {
                      "anyOf": [
                        {
                          "enum": [
                            "<本轮对象ID>"
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ],
                      "description": "Object whose grasp/contact is intentionally changed. Use null when only changing the gripper configuration without a new object target. This does not certify an empty hand; any existing carried reference is still retired."
                    },
                    "intent": {
                      "type": "string",
                      "minLength": 1
                    },
                    "selection": {
                      "type": "object",
                      "properties": {
                        "command": {
                          "enum": [
                            "open",
                            "close"
                          ]
                        },
                        "steps": {
                          "type": "integer",
                          "minimum": 1,
                          "maximum": 200
                        }
                      },
                      "required": [
                        "command"
                      ],
                      "additionalProperties": false
                    }
                  },
                  "required": [
                    "primitive",
                    "target_id",
                    "intent",
                    "selection"
                  ],
                  "additionalProperties": false
                },
                {
                  "type": "object",
                  "properties": {
                    "primitive": {
                      "const": "wait"
                    },
                    "target_id": {
                      "enum": [
                        "<本轮对象ID>"
                      ]
                    },
                    "intent": {
                      "type": "string",
                      "minLength": 1
                    },
                    "selection": {
                      "type": "object",
                      "properties": {
                        "steps": {
                          "type": "integer",
                          "minimum": 1,
                          "maximum": 200
                        }
                      },
                      "required": [],
                      "additionalProperties": false
                    }
                  },
                  "required": [
                    "primitive",
                    "target_id",
                    "intent",
                    "selection"
                  ],
                  "additionalProperties": false
                }
              ]
            },
            "minItems": 1,
            "maxItems": 12,
            "description": "Each item is a complete action: primitive, target_id, intent, selection, and reference_target_id for movement. Move points belong only inside that item’s selection.waypoints array; they are not action items or root fields."
          }
        },
        "required": [
          "visible_evidence",
          "held_object_id",
          "held_pose_relation",
          "actions"
        ],
        "additionalProperties": false
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "list_dir",
      "description": "List only published files/folders in this episode, non-recursively. No motion.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string"
          },
          "offset": {
            "type": "integer",
            "minimum": 0
          }
        },
        "required": [],
        "additionalProperties": false
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "read_text_file",
      "description": "Read a published UTF-8 file, with explicit pagination. No arbitrary run or project access.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "minLength": 1
          },
          "offset": {
            "type": "integer",
            "minimum": 0
          },
          "max_chars": {
            "type": "integer",
            "minimum": 1,
            "maximum": 40000
          }
        },
        "required": [
          "path"
        ],
        "additionalProperties": false
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "read_image",
      "description": "Optionally view an indexed published image, including views not yet in available_sources. A successful read registers its cloud view_id or native pixel source, subject to owner/time checks. latest resolves an immutable version; current local and _measured views provide current measured samples. Reading a rigid latest original-shape entry may attempt supported pose matching; the result states coordinate availability. Current context samples require explicit reference_target_id. No camera motion.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "minLength": 1
          }
        },
        "required": [
          "path"
        ],
        "additionalProperties": false
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_state",
      "description": "Read current robot and observed geometry facts. No physics or new camera motion.",
      "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": false
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "refresh_object",
      "description": "Optional object association capability. Submit exactly one tool call in this reply; do not mix it with other calls. inspect organizes saved current RGB-D into numbered fragments without camera motion, identity choice or whole-object registration. bind explicitly selects shown fragments or none and publishes current point views; it does not restore the old complete template. rebind_held_pose optionally attempts the same grasp's relative-pose recovery. initial_scene only corrects untouched initial identities. Does not certify holding or clear contact warnings.",
      "parameters": {
        "oneOf": [
          {
            "type": "object",
            "properties": {
              "mode": {
                "const": "inspect"
              },
              "object_id": {
                "enum": [
                  "<本轮对象ID>"
                ]
              },
              "source_step": {
                "type": "integer",
                "minimum": 0
              },
              "sources": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "description": "Optional current already-viewed native rectangles. Omit for the complete current dual views. Rectangles are search windows, not semantic masks.",
                "items": {
                  "type": "object",
                  "properties": {
                    "observation_id": {
                      "type": "string",
                      "minLength": 1
                    },
                    "camera": {
                      "enum": [
                        "agentview",
                        "robot0_eye_in_hand"
                      ]
                    },
                    "bounds_xyxy": {
                      "type": "array",
                      "items": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1024
                      },
                      "minItems": 4,
                      "maxItems": 4
                    }
                  },
                  "required": [
                    "observation_id",
                    "camera",
                    "bounds_xyxy"
                  ],
                  "additionalProperties": false
                }
              }
            },
            "required": [
              "mode",
              "object_id"
            ],
            "additionalProperties": false
          },
          {
            "type": "object",
            "properties": {
              "mode": {
                "const": "bind"
              },
              "object_id": {
                "enum": [
                  "<本轮对象ID>"
                ]
              },
              "source_step": {
                "type": "integer",
                "minimum": 0
              },
              "evidence_id": {
                "type": "string",
                "minLength": 1,
                "description": "A displayed current inspection ID; initial_scene is only for correcting untouched initial G identities."
              },
              "candidate_ids": {
                "type": "array",
                "items": {
                  "type": "string",
                  "minLength": 1
                },
                "uniqueItems": true
              },
              "decision": {
                "enum": [
                  "select",
                  "none"
                ]
              },
              "selection_scope": {
                "enum": [
                  "whole_visible_object",
                  "visible_part"
                ]
              },
              "point_set_scope": {
                "enum": [
                  "one_rigid_object",
                  "mixed_or_unknown"
                ],
                "description": "Whether ALL selected points belong to one rigid object, even for a visible part. Mixed or uncertain groups remain selectable for localization but cannot define a carried shape. Omission means mixed_or_unknown; a tight current inspect rectangle can isolate a clean subset."
              },
              "rebind_held_pose": {
                "type": "boolean",
                "description": "Optional explicit visual claim that this same object remains held but its pose relative to the hand changed. Suspends the old relation, then uses a clean one_rigid_object current patch to register the original frozen samples in this same call. Mixed or unsupported matches remain suspended. Does not certify holding or clear contact warnings. Default false."
              },
              "description": {
                "type": "string",
                "minLength": 1
              },
              "visible_evidence": {
                "type": "string",
                "minLength": 1
              }
            },
            "required": [
              "mode",
              "object_id",
              "evidence_id",
              "candidate_ids",
              "decision",
              "visible_evidence"
            ],
            "additionalProperties": false,
            "allOf": [
              {
                "if": {
                  "properties": {
                    "decision": {
                      "const": "select"
                    }
                  }
                },
                "then": {
                  "required": [
                    "selection_scope",
                    "description"
                  ],
                  "properties": {
                    "candidate_ids": {
                      "minItems": 1
                    }
                  }
                },
                "else": {
                  "properties": {
                    "candidate_ids": {
                      "maxItems": 0
                    }
                  }
                }
              }
            ]
          }
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "finish",
      "description": "Submit exactly one tool call in this reply; do not mix it with other calls. Stop with a visible-evidence assessment. blocked/failure are valid. The benchmark is evaluated separately.",
      "parameters": {
        "type": "object",
        "properties": {
          "status": {
            "enum": [
              "success",
              "failure",
              "blocked"
            ]
          },
          "summary": {
            "type": "string",
            "minLength": 1
          },
          "goal_evidence": {
            "type": "array",
            "minItems": 1,
            "items": {
              "type": "object",
              "properties": {
                "goal": {
                  "type": "string",
                  "minLength": 1,
                  "description": "Verbatim original_task text. At least one entry must quote the ENTIRE original_task; optional extra entries may quote its phrases. Put your interpretation separately."
                },
                "interpretation": {
                  "type": "string",
                  "minLength": 1,
                  "description": "Why the observed object/part corresponds to that original phrase; separate from whether the physical relation was achieved."
                },
                "assessment": {
                  "enum": [
                    "satisfied",
                    "contradicted",
                    "unknown"
                  ]
                },
                "image_paths": {
                  "type": "array",
                  "items": {
                    "type": "string",
                    "minLength": 1
                  },
                  "minItems": 1
                },
                "visible_evidence": {
                  "type": "string",
                  "minLength": 1
                }
              },
              "required": [
                "goal",
                "assessment",
                "image_paths",
                "visible_evidence"
              ],
              "additionalProperties": false
            }
          },
          "task_referents": {
            "type": "array",
            "maxItems": 8,
            "description": "Optional updates of task-part interpretations, keyed by object_id and a verbatim original-task phrase. Retained independently of route notes. Omit unchanged entries; do not solve unrelated future parts now.",
            "items": {
              "type": "object",
              "properties": {
                "object_id": {
                  "enum": [
                    "<本轮对象ID>"
                  ]
                },
                "task_phrase": {
                  "type": "string",
                  "minLength": 1,
                  "maxLength": 240
                },
                "status": {
                  "enum": [
                    "selected",
                    "unresolved"
                  ]
                },
                "interpretation": {
                  "type": "string",
                  "minLength": 1,
                  "maxLength": 600
                },
                "image_paths": {
                  "type": "array",
                  "items": {
                    "type": "string",
                    "minLength": 1
                  },
                  "minItems": 1,
                  "maxItems": 4
                },
                "revision_basis": {
                  "type": "string",
                  "minLength": 1,
                  "maxLength": 600
                }
              },
              "required": [
                "object_id",
                "task_phrase",
                "status"
              ],
              "additionalProperties": false,
              "allOf": [
                {
                  "if": {
                    "properties": {
                      "status": {
                        "const": "selected"
                      }
                    }
                  },
                  "then": {
                    "required": [
                      "interpretation",
                      "image_paths"
                    ]
                  }
                }
              ]
            }
          }
        },
        "required": [
          "status",
          "summary",
          "goal_evidence"
        ],
        "additionalProperties": false
      }
    }
  }
]
```

</details>

<details>
<summary>观察接口schema生成函数，来源scene_catalog.py:185</summary>

```python
def binding_schema(catalog):
    text={'type':'string','minLength':1}
    source=obj({'observation_id':{'enum':sorted({v['observation_id'] for v in catalog['table_sources']})},
        'camera':{'enum':['agentview','robot0_eye_in_hand']},
        'bounds_xyxy':{'type':'array','items':{'type':'integer','minimum':0,'maximum':1024},
            'minItems':4,'maxItems':4}},['observation_id','camera','bounds_xyxy'])
    item=obj({'object_id':text,'component_ids':{'type':'array','items':{'enum':sorted(catalog['components'])},
        'uniqueItems':True},'source_regions':{'type':'array','items':source,'minItems':1,
            'description':'Optional coarse native-image regions only when G groups omit an essential part. Not contact points.'},'description':text,
        'interactions':{'type':'array','items':text,'minItems':1},
        'shape_motion':{'enum':['rigid','articulated','unknown'],
            'description':'Optional claim about this referenced point-set scope, not its name. rigid keeps shape and changes pose; articulated has relative part motion; omission is unknown.'},
        'required_surfaces':{'type':'array','items':{'enum':['overall_shape','contact_surface','opening','interior_support']},
            'minItems':1,'uniqueItems':True},
        'missing_surfaces':{'type':'array','items':{'enum':['overall_shape','contact_surface','opening','interior_support']},
            'uniqueItems':True,'description':'Surfaces that still need a complementary real view. Empty means reuse baseline for A review.'}},
        ['object_id','component_ids','description','interactions','required_surfaces'])
    item['anyOf']=[{'properties':{'component_ids':{'minItems':1}}},
        {'required':['source_regions']}]
    return obj({'source_step':{'const':catalog['source_step']},
        'objects':{'type':'array','items':item,'minItems':1}},['source_step','objects'])
```

</details>

## 4．程序附言和几何字段释义

以下是清理前实际请求中已出现、在清理后源码中仍逐字存在的固定释义；动态机器人状态、对象ID及数值由本轮生成。错误分支及未出现的动态反馈仍以本轮实际日志为准。完整源文件与旧审阅均保留。

<details>
<summary>展开固定附言原文及当前来源</summary>


来源：[src/sketchik/cloud_extents.py:61](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/cloud_extents.py:61)

```text
observed points above table threshold
```


来源：[src/sketchik/cloud_selection.py:221](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/cloud_selection.py:221)

```text
Frozen point table of this view only; G numbers are not persistent object identities across repackaging or source steps.
```


来源：[src/sketchik/current_geometry.py:115](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/current_geometry.py:115)

```text
Known source-state robot geometry only. Near-robot object samples can also be excluded; inspect raw RGB. Not a collision check.
```


来源：[src/sketchik/current_geometry.py:119](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/current_geometry.py:119)

```text
Narrow measured tabletop band only; raw points stay saved. Not a semantic floor or cavity detector.
```


来源：[src/sketchik/current_geometry.py:120](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/current_geometry.py:120)

```text
Current native-image rectangles are search windows; G groups are visible geometric fragments, not object identities.
```


来源：[src/sketchik/episode_state.py:136](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/episode_state.py:136)

```text
A interpretation of actual images; not program holding certification
```


来源：[src/sketchik/grounding_packet.py:441](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/grounding_packet.py:441)

```text
Measured visible horizontal surfaces in the operation ROI; no semantic floor/roof identity or usable cavity clearance certified
```


来源：[src/sketchik/grounding_packet.py:452](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/grounding_packet.py:452)

```text
Connected observed fragments in the operation ROI. A group may contain multiple touching surfaces; one object may occupy multiple groups. Bounds are visible extents, not usable cavity dimensions.
```


来源：[src/sketchik/hand_observation.py:130](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/hand_observation.py:130)

```text
current observed non-robot near-hand surfaces; identities unassigned
```


来源：[src/sketchik/hand_observation.py:139](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/hand_observation.py:139)

```text
eef_propagated_observed_shape_hypothesis
```


来源：[src/sketchik/hand_observation.py:153](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/hand_observation.py:153)

```text
current actual finger opening projected at the recorded goal, not historical execution
```


来源：[src/sketchik/hand_observation.py:163](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/hand_observation.py:163)

```text
Orange: hypothetical known robot parts. Cyan: supplied associated held-point hypothesis only. Projection/envelope overlap does not certify collision or attachment.
```


来源：[src/sketchik/hand_observation.py:355](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/hand_observation.py:355)

```text
EEF world-centred XY +/-0.22 m, Z +/-0.25 m
```


来源：[src/sketchik/hand_observation.py:360](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/hand_observation.py:360)

```text
Retained observed shape under the current grasp hypothesis, transformed by EEF. Actual RGB slip or separation overrides this prediction.
```


来源：[src/sketchik/held_pose_review.py:24](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/held_pose_review.py:24)

```text
actual_close_event_not_certified_holding
```


来源：[src/sketchik/held_pose_review.py:28](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/held_pose_review.py:28)

```text
Compare actual object BODY pose relative to the pads in labelled previous/current images. Camera/hand rotation alone is not object slip. Holding and unchanged relative pose are separate claims. Predictions and jaw gap alone cannot certify unchanged; use unknown when comparison is insufficient.
```


来源：[src/sketchik/held_pose_review.py:51](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/held_pose_review.py:51)

```text
Previous actual RGB at the last A held claim or actual closing event; not current geometry or certified grasp baseline
```


来源：[src/sketchik/image_axes.py:125](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/image_axes.py:125)

```text
orthographic camera right/down; direction only
```


来源：[src/sketchik/libero.py:168](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/libero.py:168)

```text
top-left pixel origin; camera x right, y down, z forward; depth metres
```


来源：[src/sketchik/object_registration.py:183](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/object_registration.py:183)

```text
insufficient_current_surface_overlap
```


来源：[src/sketchik/object_registration.py:342](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/object_registration.py:342)

```text
nearer_other_initial_instance_identity_is_ambiguous
```


来源：[src/sketchik/object_registration.py:420](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/object_registration.py:420)

```text
Observed workspace XY plus hand/template extent when provided; objects outside this search remain unknown.
```


来源：[src/sketchik/object_registration.py:425](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/object_registration.py:425)

```text
Thresholded rigid pose estimate of earlier observed samples, not new observation, holding or semantic certification. Hidden surfaces remain unknown.
```


来源：[src/sketchik/observed_clearance.py:36](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/observed_clearance.py:36)

```text
current_depth_with_self_and_foreground_occlusion
```


来源：[src/sketchik/observed_clearance.py:107](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/observed_clearance.py:107)

```text
conservative_convex_hull_of_observed_carried_points
```


来源：[src/sketchik/observed_clearance.py:240](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/observed_clearance.py:240)

```text
observed_reference_predicted_by_eef
```


来源：[src/sketchik/observed_dimensions.py:182](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/observed_dimensions.py:182)

```text
Observed horizontal surface within supplied ROI; support, aperture and clearance unverified
```


来源：[src/sketchik/primitives.py:119](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/primitives.py:119)

```text
max(requested_steps,60), limited by remaining episode budget
```


来源：[src/sketchik/primitives.py:122](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/primitives.py:122)

```text
Binary open/close, not a target jaw width. Fewer steps do not request a narrower gap. Settling or a nonzero gap does not certify holding.
```


来源：[src/sketchik/primitives.py:374](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/primitives.py:374)

```text
Measured error reduction only; contact cause and attachment remain unknown
```


来源：[src/sketchik/robot_geometry.py:99](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/robot_geometry.py:99)

```text
Axis projection of the full palm envelope behind the pads, not the distance to its far end alone. Negative means the envelope extends forward of the pad plane; not free space or a collision check.
```


来源：[src/sketchik/robot_geometry.py:105](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/robot_geometry.py:105)

```text
Current robot envelope about pad_midpoint, not solid occupancy or a collision check. Positive tool Z points from palm toward fingertips. Program converts the chosen reference to EEF at the requested orientation; do not add that compensation to offset_m.
```


来源：[src/sketchik/robot_geometry.py:281](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/robot_geometry.py:281)

```text
known robot collision model and actual finger joint state
```


来源：[src/sketchik/robot_geometry.py:290](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/robot_geometry.py:290)

```text
Robot mechanical envelope at this opening; not object dimensions, a grasp proposal or collision-free certification. EEF is not the fingertip or object centre.
```


来源：[src/sketchik/scene_catalog.py:91](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/scene_catalog.py:91)

```text
All observed tabletop geometry, including unbound objects and ungrouped points. Touching objects may merge, occlusion may split them; no hidden completion or collision certification.
```


来源：[src/sketchik/scene_catalog.py:175](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/scene_catalog.py:175)

```text
last_observed; unchanged pose assumed, not verified
```


来源：[src/sketchik/scene_catalog.py:182](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/scene_catalog.py:182)

```text
Historical bounds are not updated poses. A shared group is uncertain if any bound part moved. Unbound objects may also have been bumped; use current RGB.
```


来源：[src/sketchik/scene_catalog.py:255](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/scene_catalog.py:255)

```text
Components bound; program acquires evidence, A confirms semantic surfaces and first sketch next.
```


来源：[src/sketchik/single_agent.py:42](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:42)

```text
Known axes of this native camera at its recorded source_step, rounded for display only. Not object axes or a definition of task front/back. Exact calibration/point conversion stays in the program.
```


来源：[src/sketchik/single_agent.py:59](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:59)

```text
Known rendering directions, not the physical camera pose, object axes or a surface normal.
```


来源：[src/sketchik/single_agent.py:116](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:116)

```text
Offsets include X, Y and Z. These values describe the CURRENT orientation only. For a move the existing compiler recomputes the chosen reference at the TARGET orientation: EEF = anchor + offset - reference_offset. Do not add this compensation again. EEF is not the held centre or bottom. Held references use associated observed points, not verified attachment or complete shape.
```


来源：[src/sketchik/single_agent.py:130](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:130)

```text
Same compiler point set; axes ordered by sample variance, signs arbitrary, small variance gaps imply unstable directions. Not semantic axes or chosen grasp directions.
```


来源：[src/sketchik/single_agent.py:179](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:179)

```text
Current evidence only. Full robot geometry/provenance in current.json and robot_geometry.json; full receipts at receipt_path. Principal projections are observed point spans, not a solid box or a reconstructed centre. Use exact pixels; program computes reference-to-EEF compensation.
```


来源：[src/sketchik/single_agent.py:606](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:606)

```text
Whole observed object and adjacent structure; rectangles/groups are not semantic masks or complete shape.
```


来源：[src/sketchik/single_agent.py:646](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:646)

```text
No recorded owner-motion invalidation; earlier capture time alone does not invalidate an anchor.
```


来源：[src/sketchik/single_agent.py:647](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:647)

```text
Selected point source_step must be at least invalidated_at_step.
```


来源：[src/sketchik/single_agent.py:648](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:648)

```text
Recorded time/owner checks only, not semantic identity or unobserved-motion certification.
```


来源：[src/sketchik/single_agent.py:661](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:661)

```text
Earlier A observation claim; verify against actual evidence before acting
```


来源：[src/sketchik/single_agent.py:662](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:662)

```text
Observed surfaces only. Initial scans become historical after object motion.
```


来源：[src/sketchik/single_agent.py:828](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:828)

```text
G IDs are geometric fragments, not identities. Select the intended visible object/part, multiple fragments or none. No candidate has been chosen automatically.
```


来源：[src/sketchik/single_agent.py:1219](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:1219)

```text
All original points retained; observed surfaces only, not live object coordinates.
```


来源：[src/sketchik/single_agent.py:1238](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:1238)

```text
Exact saved point table; use view_id and pixel_xy in move. No separate query needed.
```


来源：[src/sketchik/single_agent.py:1387](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:1387)

```text
Current episode only; no credentials, simulator object truth, evaluator or other projects.
```


来源：[src/sketchik/single_agent.py:1554](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:1554)

```text
Intents are historical model claims, not evidence identifying task parts. Completion records commands/results only; it does not verify task meaning or fulfilment.
```


来源：[src/sketchik/single_agent.py:1652](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:1652)

```text
Historical scene/shape references. Check CURRENT owner invalidations before selecting a point. Read an indexed image once before selecting it; additional angles and full metadata remain in scene.json.
```


来源：[src/sketchik/single_agent.py:1672](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:1672)

```text
PREVIOUS actual at last held claim or actual close; not current pose or certified holding
```


来源：[src/sketchik/single_agent.py:1680](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:1680)

```text
CURRENT is the latest measured state. available_sources lists already-read images; owner/time checks still apply. Pixel + offset + orientation compiles directly. Full evidence: current.json; all readable angles: scene.json.
```


来源：[src/sketchik/single_agent.py:1692](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:1692)

```text
Latest actual robot outcome images. Each record identifies its source step. When available, PREVIOUS actual dual RGB supplies the last held-claim or close-event comparison; compare the object against pads and fixed scene across the actual motion, not against a propagated prediction. The structured CURRENT/checkpoint gives current numeric state; older scene references remain historical.
```


来源：[src/sketchik/single_agent.py:1713](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:1713)

```text
A task-part interpretation, not verified identity, current coordinates or geometric fit. Source images are historical; motion invalidations still apply. Only new identity/structural evidence revises this interpretation.
```


来源：[src/sketchik/single_agent.py:1790](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:1790)

```text
Execution phase request/tool/batch limits; physical steps cover the whole episode including acquisition. Each attempted tool call consumes a tool slot; over-budget replies are rejected before any dispatch.
```


来源：[src/sketchik/single_agent.py:1861](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:1861)

```text
Complete acquisition conversation, historical robot states and model judgments; not current evidence.
```


来源：[src/sketchik/single_agent.py:1882](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:1882)

```text
Raw acquisition telemetry is historical. All native images and full source data remain readable; no camera motion.
```


来源：[src/sketchik/single_agent.py:1897](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/single_agent.py:1897)

```text
Distinct supplemental wrist views before reset, not current outcomes. All other acquisition images/numeric telemetry remain in acquisition/index.json and observations/.
```


来源：[src/sketchik/stage_context.py:90](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/stage_context.py:90)

```text
serialized text characters with image payloads hashed; not a tokenizer
```


来源：[src/sketchik/stage_context.py:246](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/stage_context.py:246)

```text
Latest single_A_direct_execution CURRENT message and current.json
```


来源：[src/sketchik/stage_context.py:255](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/stage_context.py:255)

```text
Explicit working-context transition within this same A episode. Older complete dialogues remain archived; the recent complete dialogues above are retained verbatim. The structured state below is program evidence, not a model reasoning transcript or task-success certification. Continue from current real outcome, never replay completed actions. Available interfaces are only those in the current tool schema. Old observations and known robot calibration remain in the episode archive; use the available history/file tools to retrieve them when needed.
```


来源：[src/sketchik/step_recovery.py:174](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/step_recovery.py:174)

```text
No remaining T1 allowance or no measured recent progress; do not repeat a stalled command
```


来源：[src/sketchik/surface_evidence.py:32](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/surface_evidence.py:32)

```text
approximate_visible_near_horizontal_pixels_only
```


来源：[src/sketchik/task_execution.py:74](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/task_execution.py:74)

```text
Recorded robot commands/results from the preceding stage; causes and holding are not certified. Do not repeat a stalled goal as a fresh route. Historical recovery usage does not consume the current request allowance.
```


来源：[src/sketchik/task_execution.py:726](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/task_execution.py:726)

```text
Local recovery failed; suffix not executed
```


来源：[src/sketchik/task_execution.py:749](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/task_execution.py:749)

```text
Gripper action changed the previous grasp reference
```


来源：[src/sketchik/task_trial.py:31](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/task_trial.py:31)

```text
Saved with native RGB-D; the program performs coordinate transforms.
```


来源：[src/sketchik/task_trial.py:40](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/task_trial.py:40)

```text
WORLD compass shown in this image; direction only, no point or metric scale.
```


来源：[src/sketchik/task_trial.py:237](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/task_trial.py:237)

```text
Use individual finger lengths/thickness, pad offsets, current jaw gap and the entire palm/crossbar envelope when choosing grasp manner, approach, insertion and withdrawal. The current tool schema defines whether you submit semantics or actual parameters. All dimensions are in EEF tool axes and change world orientation with the hand. The palm may collide even when fingertips fit.
```


来源：[src/sketchik/task_trial.py:263](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/task_trial.py:263)

```text
Only the following submit_scene_requirements request; expires when the acquisition batch ends. Retained later as historical context.
```


来源：[src/sketchik/tools.py:187](/media/wsqlab/more/kaptainiii/projects/SketchIK/src/sketchik/tools.py:187)

```text
Historical RGB, not a new observation
```


</details>

## 5．系统追加与网关工具格式

来源：src/sketchik/task_trial.py

```text
本轮由用户明确选择IK-only对照：保留完整位姿IK、关节限位、路径IK和严格FK，关闭观测表面碰撞筛查及其绕行。点云仍用于定位和持物参考，不把无碰撞拒绝或运动收敛当成成功；以实际新图复核抓持、放置及关门。
```

来源：.runtime/Flowith2API/internal/service/tool.go；此模板追加当前工具schema及XML调用格式，不含任务策略。

```text
You have access to the following functions. Use them if required:

%s

If a function should be called, return exactly one function_calls XML block.
Include each intended invoke once; do not repeat calls or emit draft blocks.
Follow each tool's limit on mixing calls. Object/array parameters must contain
complete valid JSON with balanced braces/brackets, without outer quotes or Markdown fences.

XML form:
<function_calls>
<invoke name="function_name">
<parameter name="param1">value1</parameter>
</invoke>
</function_calls>

If no function is needed, respond normally.
```

## 6．保留的数据与本轮配置

原始1024双RGB-D、点表、标定、历史图、源码和视频保持。API传输JPEG95已获用户授权；同一A保持完整源历史，工作窗口按公开策略处理，旧图按文件读回，无长期memory。当前用户配置Sol low、65536额度；提供方是否完整执行这些参数不能仅凭本地配置认证。

基础扫描、补拍、尺寸和点云恢复保留；整任务固定IK-only，真实IK/FK及5毫米操作容差保持。每轮从完整初态独立记录，源码哈希及实际结果随运行归档。T1有既有接触基线时逐步核对机器人实测夹爪间隙与接触边界；扫描中的程序异常直接停止，不降格为普通取景不足。


## 7．可选异步阶段（协议73，失败时取消在途规划）

当前段正常完成后直接执行同A保存的下一段完整草图，包括抓取、放置、开闭爪和活动结构。没有逐段accept/replan审批阶段或视觉确认开关。语义未知记为未认证，不阻断正常接续、不制造A持物/图像判断。实际反馈随上下文进入后续规划。

仍保留候选哈希/所属运行/批次/请求/物理步/一次消费、实际完成和接触/当前深度矛盾检查。实际用到的点源或held参考必须经当前绑定和真实IK/FK；新抓持epoch按真实夹爪事件更新，不能要求与准备时相同，也不能恢复旧epoch坐标。程序来源单列，不增加A请求/工具次数、不调用apply_assessment。默认同步和IK-only不变。

实际失败：立即退役旧候选并中断仍在生成的HTTP请求，等待连接及线程清理后由同A恢复，不等待旧草图完整生成。完整返回的旧候选只留作不可执行参考；不完整输出保留原始流和取消记录，不伪造工具回复。主动取消不重试、不算格式错误，实际API尝试与已知用量仍计数；上游停止计费未认证。同A依据真实停止状态恢复，恢复段执行期间准备新后续段；不拼接旧后缀、不复用失效点、不重放完成动作。

终态：只有当前正在执行的草图预计结束整个任务时，A可提交finish_review。这只是段末回到正常7工具的提议，不是成功事实或新的机器人动作；同A根据真实反馈finish或继续修复。不为终态检查强制生成wait。

以下为阶段工具实际schema（动作复用7工具中的既有定义）：

```json
{
  "type": "function",
  "function": {
    "name": "draft_next_sketch",
    "description": "Save the complete next-subtask sketch using existing geometry; normal completed execution continues it directly, including gripper and articulated actions. Preconditions are planning notes, not an extra approval gate. Future outcomes remain unknown. Only when the executing sketch is expected to finish the entire task, use finish_review to request normal tools on the actual outcome for finish or repair. This is not defer or a success declaration; do not invent wait actions. No robot or file tools execute in this phase.",
    "parameters": {
      "type": "object",
      "properties": {
        "status": {
          "enum": [
            "ready",
            "finish_review"
          ]
        },
        "reason": {
          "type": "string",
          "minLength": 1
        },
        "preconditions": {
          "type": "string",
          "minLength": 1
        },
        "intent": {
          "type": "string",
          "minLength": 1
        },
        "actions": {
          "type": "array",
          "items": {
            "oneOf": [
              {
                "type": "object",
                "properties": {
                  "primitive": {
                    "const": "move"
                  },
                  "target_id": {
                    "enum": [
                      "<本轮对象ID>"
                    ]
                  },
                  "intent": {
                    "type": "string",
                    "minLength": 1
                  },
                  "selection": {
                    "type": "object",
                    "properties": {
                      "waypoints": {
                        "type": "array",
                        "items": {
                          "type": "object",
                          "properties": {
                            "anchor": {
                              "type": "object",
                              "properties": {
                                "kind": {
                                  "enum": [
                                    "current_eef",
                                    "cloud_point",
                                    "image_point"
                                  ]
                                },
                                "view_id": {
                                  "type": "string",
                                  "minLength": 1
                                },
                                "group_id": {
                                  "type": "integer",
                                  "minimum": 0
                                },
                                "observation_id": {
                                  "type": "string"
                                },
                                "camera": {
                                  "enum": [
                                    "agentview",
                                    "robot0_eye_in_hand"
                                  ]
                                },
                                "pixel_xy": {
                                  "type": "array",
                                  "items": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 1023
                                  },
                                  "minItems": 2,
                                  "maxItems": 2
                                }
                              },
                              "required": [
                                "kind"
                              ],
                              "additionalProperties": false,
                              "description": "Observed surface point, resolved exactly from the saved pixel table at execution; do not estimate its XYZ. current_eef uses the actual EEF at each waypoint entry; ordinary relative EEF translations pair current_eef with reference=eef."
                            },
                            "reference": {
                              "enum": [
                                "eef",
                                "pad_0",
                                "pad_1",
                                "pad_midpoint",
                                "held_bottom",
                                "held_center"
                              ],
                              "description": "Point placed at anchor+offset: eef = grip_site origin; pad_midpoint = midpoint between the two calibrated finger-pad centres; pad_0/pad_1 = one pad centre. Program handles the entire reference-to-EEF compensation at the requested orientation. Do not add that compensation again to offset_m. Point to align with anchor+offset. held_center is the world-axis AABB centre of the associated observed carried points after applying the TARGET orientation; held_bottom uses that same XY centre and min Z, not a single lowest point with unknown XY. The program compensates the EEF offset. These are observed-shape references, not complete object geometry."
                            },
                            "offset_m": {
                              "type": "array",
                              "items": {
                                "type": "number"
                              },
                              "minItems": 3,
                              "maxItems": 3,
                              "description": "Desired displacement in metres from the observed anchor to the chosen reference point, not to EEF unless reference=eef."
                            },
                            "offset_frame": {
                              "enum": [
                                "world",
                                "tool"
                              ],
                              "description": "world uses world XYZ; tool uses the TARGET tool axes, not the current axes."
                            },
                            "orientation": {
                              "anyOf": [
                                {
                                  "const": "preserve"
                                },
                                {
                                  "type": "object",
                                  "properties": {
                                    "rpy_world_rad": {
                                      "type": "array",
                                      "items": {
                                        "type": "number"
                                      },
                                      "minItems": 3,
                                      "maxItems": 3
                                    }
                                  },
                                  "required": [
                                    "rpy_world_rad"
                                  ],
                                  "additionalProperties": false
                                },
                                {
                                  "type": "object",
                                  "properties": {
                                    "rpy_delta_tool_rad": {
                                      "type": "array",
                                      "items": {
                                        "type": "number"
                                      },
                                      "minItems": 3,
                                      "maxItems": 3
                                    }
                                  },
                                  "required": [
                                    "rpy_delta_tool_rad"
                                  ],
                                  "additionalProperties": false
                                },
                                {
                                  "type": "object",
                                  "properties": {
                                    "finger_direction": {
                                      "type": "array",
                                      "items": {
                                        "type": "number"
                                      },
                                      "minItems": 3,
                                      "maxItems": 3
                                    },
                                    "closing_axis": {
                                      "type": "array",
                                      "items": {
                                        "type": "number"
                                      },
                                      "minItems": 3,
                                      "maxItems": 3
                                    }
                                  },
                                  "required": [
                                    "finger_direction",
                                    "closing_axis"
                                  ],
                                  "additionalProperties": false
                                }
                              ],
                              "description": "finger_direction = target tool +Z in world, closing_axis = target tool +X in world. Program computes the full quaternion. preserve retains all current axes."
                            }
                          },
                          "required": [
                            "anchor",
                            "reference",
                            "offset_m",
                            "offset_frame",
                            "orientation"
                          ],
                          "additionalProperties": false
                        },
                        "minItems": 1,
                        "maxItems": 6,
                        "description": "Ordered full-pose waypoints of this move. Gripper changes are separate actions in the enclosing sketch."
                      }
                    },
                    "required": [
                      "waypoints"
                    ],
                    "additionalProperties": false
                  },
                  "reference_target_id": {
                    "enum": [
                      "<本轮对象ID>",
                      "stage_support"
                    ],
                    "description": "Owner of the observed anchor. May be omitted only when all move points are cloud_point views of one known object; the program then uses that registered owner. Native pixels, relative EEF moves and rotations require an explicit reference."
                  }
                },
                "required": [
                  "primitive",
                  "target_id",
                  "intent",
                  "selection"
                ],
                "additionalProperties": false
              },
              {
                "type": "object",
                "properties": {
                  "primitive": {
                    "const": "rotate_roll"
                  },
                  "target_id": {
                    "enum": [
                      "<本轮对象ID>"
                    ]
                  },
                  "intent": {
                    "type": "string",
                    "minLength": 1
                  },
                  "selection": {
                    "type": "object",
                    "properties": {
                      "angle_rad": {
                        "type": "number"
                      },
                      "settle_steps": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 200
                      }
                    },
                    "required": [
                      "angle_rad"
                    ],
                    "additionalProperties": false
                  },
                  "reference_target_id": {
                    "enum": [
                      "<本轮对象ID>",
                      "stage_support"
                    ],
                    "description": "Owner of the observed anchor. May be omitted only when all move points are cloud_point views of one known object; the program then uses that registered owner. Native pixels, relative EEF moves and rotations require an explicit reference."
                  }
                },
                "required": [
                  "primitive",
                  "target_id",
                  "intent",
                  "selection",
                  "reference_target_id"
                ],
                "additionalProperties": false
              },
              {
                "type": "object",
                "properties": {
                  "primitive": {
                    "const": "rotate_pitch"
                  },
                  "target_id": {
                    "enum": [
                      "<本轮对象ID>"
                    ]
                  },
                  "intent": {
                    "type": "string",
                    "minLength": 1
                  },
                  "selection": {
                    "type": "object",
                    "properties": {
                      "angle_rad": {
                        "type": "number"
                      },
                      "settle_steps": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 200
                      }
                    },
                    "required": [
                      "angle_rad"
                    ],
                    "additionalProperties": false
                  },
                  "reference_target_id": {
                    "enum": [
                      "<本轮对象ID>",
                      "stage_support"
                    ],
                    "description": "Owner of the observed anchor. May be omitted only when all move points are cloud_point views of one known object; the program then uses that registered owner. Native pixels, relative EEF moves and rotations require an explicit reference."
                  }
                },
                "required": [
                  "primitive",
                  "target_id",
                  "intent",
                  "selection",
                  "reference_target_id"
                ],
                "additionalProperties": false
              },
              {
                "type": "object",
                "properties": {
                  "primitive": {
                    "const": "rotate_yaw"
                  },
                  "target_id": {
                    "enum": [
                      "<本轮对象ID>"
                    ]
                  },
                  "intent": {
                    "type": "string",
                    "minLength": 1
                  },
                  "selection": {
                    "type": "object",
                    "properties": {
                      "angle_rad": {
                        "type": "number"
                      },
                      "settle_steps": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 200
                      }
                    },
                    "required": [
                      "angle_rad"
                    ],
                    "additionalProperties": false
                  },
                  "reference_target_id": {
                    "enum": [
                      "<本轮对象ID>",
                      "stage_support"
                    ],
                    "description": "Owner of the observed anchor. May be omitted only when all move points are cloud_point views of one known object; the program then uses that registered owner. Native pixels, relative EEF moves and rotations require an explicit reference."
                  }
                },
                "required": [
                  "primitive",
                  "target_id",
                  "intent",
                  "selection",
                  "reference_target_id"
                ],
                "additionalProperties": false
              },
              {
                "type": "object",
                "properties": {
                  "primitive": {
                    "const": "set_gripper"
                  },
                  "target_id": {
                    "anyOf": [
                      {
                        "enum": [
                          "<本轮对象ID>"
                        ]
                      },
                      {
                        "type": "null"
                      }
                    ],
                    "description": "Object whose grasp/contact is intentionally changed. Use null when only changing the gripper configuration without a new object target. This does not certify an empty hand; any existing carried reference is still retired."
                  },
                  "intent": {
                    "type": "string",
                    "minLength": 1
                  },
                  "selection": {
                    "type": "object",
                    "properties": {
                      "command": {
                        "enum": [
                          "open",
                          "close"
                        ]
                      },
                      "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200
                      }
                    },
                    "required": [
                      "command"
                    ],
                    "additionalProperties": false
                  }
                },
                "required": [
                  "primitive",
                  "target_id",
                  "intent",
                  "selection"
                ],
                "additionalProperties": false
              },
              {
                "type": "object",
                "properties": {
                  "primitive": {
                    "const": "wait"
                  },
                  "target_id": {
                    "enum": [
                      "<本轮对象ID>"
                    ]
                  },
                  "intent": {
                    "type": "string",
                    "minLength": 1
                  },
                  "selection": {
                    "type": "object",
                    "properties": {
                      "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200
                      }
                    },
                    "required": [],
                    "additionalProperties": false
                  }
                },
                "required": [
                  "primitive",
                  "target_id",
                  "intent",
                  "selection"
                ],
                "additionalProperties": false
              }
            ]
          },
          "minItems": 1,
          "maxItems": 12,
          "description": "Each item is a complete action: primitive, target_id, intent, selection, and reference_target_id for movement. Move points belong only inside that item’s selection.waypoints array; they are not action items or root fields."
        }
      },
      "required": [
        "status",
        "reason"
      ],
      "additionalProperties": false,
      "allOf": [
        {
          "if": {
            "properties": {
              "status": {
                "const": "ready"
              }
            }
          },
          "then": {
            "required": [
              "preconditions",
              "intent",
              "actions"
            ]
          },
          "else": {
            "not": {
              "anyOf": [
                {
                  "required": [
                    "preconditions"
                  ]
                },
                {
                  "required": [
                    "intent"
                  ]
                },
                {
                  "required": [
                    "actions"
                  ]
                }
              ]
            }
          }
        }
      ]
    }
  }
}
```

Sol low/65536、原始图/精确点表、同一步复用和后台总图、原手图画法、完整源历史均保留。协议71旧轮官方true但自动0次/549.177秒保留原档，不重解释为本版在线结果。用户随后重新授权继续，协议72单轮完整初态Task0官方true，3次自动派发、1次实际运动失败由同A恢复，操作阶段8次A请求；原始664.576秒含登录过期暂停282.281秒。源码/提示与离线冻结哈希一致，完整结果见.runtime/task0_async72_run_review.json；单轮不认证稳定提速。
