# Robot Control Skill - 机械臂控制

控制 ManiSkill-3 机械臂仿真环境，通过 HTTP API 与仿真服务端交互。

> ⚠️ **重要规则**:
> - 只能运行下方列出的脚本完成任务
> - **禁止生成或修改任何代码**
> - 仿真服务端必须运行在 `http://127.0.0.1:8000`

---

## 🚀 快速启动（新对话必读）

### 前置检查
```bash
# 检查仿真服务端是否运行
cd skill/scripts
uv run python health.py
```
如果仿真服务端未运行，**提醒用户手动启动**，不要私自启动。

---

## 📋 动作指令规范

### 7 维动作向量 `[x, y, z, roll, pitch, yaw, gripper]`

| 参数 | 范围 | 说明 |
|------|------|------|
| `x` | -0.4 ~ 0.2 | 前后（米，世界坐标系） |
| `y` | -0.3 ~ 0.3 | 左右（米，世界坐标系） |
| `z` | 0 ~ 0.6 | 上下（米，世界坐标系） |
| `roll` | **-3.14** | 固定值 |
| `pitch` | **0** | 固定值 |
| `yaw` | 物体 yaw 值 | 可变值 |
| `gripper` | 1.0/0.0 | 1.0=打开，-1.0=闭合 |

> ⚠️ **关键约束**: `roll` 全程保持 `-3.14`，`pitch` 全程保持 `0`

---

## 🧭 方向坐标系

| 方向 | 轴   | 正方向 |
| ---- | ---- | ------ |
| 前   | x 轴 | +x     |
| 后   | x 轴 | -x     |
| 左   | y 轴 | +y     |
| 右   | y 轴 | -y     |
| 上   | z 轴 | +z     |
| 下   | z 轴 | -z     |

## 🎯 标准工作流

​	获取任务说明：uv run python get_task_instruction.py，按照获取的任务的说明完成任务。

### 技能1：看看桌上有什么东西

```python
~/anaconda3/bin/conda run -n yolo26 python object_detect.py
```

### 技能2：获取物体世界坐标

```python
# 获取物体世界坐标
~/anaconda3/bin/conda run -n yolo26 python get_object_pose.py
```

### 技能3：重置环境

```bash
uv run python reset.py
```

### 技能4：抓取物体 A

```bash
# === 第 1 步：获取物体坐标 ===
# 获取物体 A 坐标
~/anaconda3/bin/conda run -n yolo26 python get_object_pose.py
# 记录输出中的 [x, y, z, roll, pitch, yaw]

# === 第 2 步：抓取物体 A ===
# 2.1 移动到物体 A 上方（z+0.2 米安全高度）
uv run python action.py <Ax> <Ay> <Az+0.2> -3.14 0 <A_yaw> 1.0

# 2.2 下降到物体 A 处
uv run python action.py <Ax> <Ay> <Az> -3.14 0 <A_yaw> 1.0

# 2.3 闭合夹爪
uv run python action.py <Ax> <Ay> <Az> -3.14 0 <A_yaw> -1.0
# 等待零点几秒，确保物体被抓住
# 2.4 抬起到安全位置（z+0.2 米）
uv run python action.py <Ax> <Ay> <Az+0.2> -3.14 0 <A_yaw> -1.0
```

### 技能5：把手上的物体A放到B处

```bash
# === 第 1 步：获取物体坐标 ===
# 1.1 如果是放在桌面，Bz = 0，其他坐标自行判断
# 1.2 如果是放在和物体B相关的位置,需获取物体B位置
~/anaconda3/bin/conda run -n yolo26 python get_object_pose.py
# 记录输出中的 [x, y, z, roll, pitch, yaw],然后根据用户命令判断进一步要放在哪里

# === 第 2 步：放下物体到B处 ===
# 2.1 移动到位姿 B 上方（z+0.2 米安全高度）
B物体厚度:Bd = 2*Bz
uv run python action.py <Bx> <By> <Bd+0.2> -3.14 0 <A_yaw> -1.0

# 2.2 下降到位姿 B 处
uv run python action.py <Bx> <By> <Bd+ 0.01> -3.14 0 <A_yaw> -1.0

# 2.3 松开夹爪
uv run python action.py <Bx> <By> <Bd + 0.01> -3.14 0 <A_yaw> 1.0

# 2.4 抬起到安全位置（z+0.2 米）
uv run python action.py <Bx> <By> <Bd+0.2> -3.14 0 <A_yaw> 1.0

# 注意：如果是堆叠物体，下降的z是(下面所有的物体的厚度和+0.01)
```
