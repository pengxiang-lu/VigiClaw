# VigiClaw

VigiClaw 是一个基于 [ManiSkill](https://github.com/haosulab/ManiSkill) 的机械臂仿真与控制项目。项目将自定义 OpenClaw 任务注册到 ManiSkill/Gymnasium，并通过 FastAPI 提供 HTTP 控制接口，配套 Skill 脚本可以完成环境检查、获取任务说明、读取相机数据、检测物体、发送机械臂动作和重置环境。

## 功能

- 基于 ManiSkill 的 Panda 机械臂仿真环境
- OpenClaw 自定义任务，例如 `OpenclawMoveApple`、`OpenclawPickCube`
- 7 维末端执行器控制：`[x, y, z, roll, pitch, yaw, gripper]`
- FastAPI HTTP 服务
- RGB、深度图像和相机内外参读取
- 物体检测与视觉定位脚本
- 任务评估、任务统计和 CSV 结果记录

## OpenClaw 任务列表

项目通过 `openclaw_task` 注册了以下 OpenClaw 任务：

| 任务名称 | 任务简介 |
| --- | --- |
| `OpenclawPickCube` | 抓取红色方块，并将其提起到指定高度。 |
| `OpenclawPickCup` | 从两个颜色不同但形状相同的杯子中，抓取指定目标杯子并提起。 |
| `OpenclawPickBanana` | 抓取香蕉，并将其提起到指定高度。 |
| `OpenclawMoveApple` | 抓取苹果，按照随机指定的方向移动一定距离后放下。 |
| `OpenclawPutBananaOnPlate` | 抓取香蕉并将其放到盘子上。 |
| `OpenclawStackCubes` | 按顺序将红色方块堆叠到绿色方块上、蓝色方块上，最后将黄色方块堆叠到蓝色方块上。 |
| `OpenclawDailyScene` | 在包含棒球、香蕉、杯子、苹果和马克笔的日常场景中，根据指令抓取工具（默认指马克笔）。 |
| `OpenclawPutTogether` | 将红、绿、蓝三个方块分别放入对应颜色的碗中。 |
| `OpenclawStackCubesDisturb` | 完成红、绿、蓝方块的堆叠，并将蓝色方块堆叠到红色方块上；任务过程中可能发生方块干扰。 |
| `OpenclawPutTogetherDisturb` | 将方块放入对应颜色的碗中；任务过程中可能出现方块回移或碗移动等干扰。 |
| `OpenclawOpenCabinet` | 打开柜子，并抓取柜子内部的物体。 |
| `OpenclawPushCube` | 推动目标方块，按照随机指定的方向移动一定距离。 |
| `OpenclawObstacleAvoid` | 绕过蓝色障碍物，抓取红色方块并将其提起。 |

## 环境要求

- Linux 或其他支持 SAPIEN/ManiSkill 的系统
- Python `>=3.10,<3.13`
- 可用的 GPU/图形环境（使用 `render_mode="human"` 时需要）
- ManiSkill `>=3.0.1`

## 安装

### 使用 uv（推荐）

```bash
git clone https://github.com/<your-name>/VigiClaw.git
cd VigiClaw

uv venv
uv sync
source .venv/bin/activate
```

### 使用 pip

```bash
git clone https://github.com/<your-name>/VigiClaw.git
cd VigiClaw

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

视觉 Skill 脚本还需要额外依赖，按实际使用的脚本安装：

```bash
pip install requests numpy Pillow opencv-python ultralytics
```

### 安装YOLO 环境

本项目的物体检测和视觉定位 Skill 使用 Ultralytics YOLO，推荐使用名为 `yolo26` 的 Conda 环境：

```bash
conda create -n yolo26 python=3.10 -y
conda activate yolo26
pip install requests numpy Pillow opencv-python ultralytics
```

如果 `yolo26` 环境已经存在，只需要执行：

```bash
conda activate yolo26
pip install requests numpy Pillow opencv-python ultralytics
```

运行视觉脚本前，请确保已经激活该环境：

```bash
conda activate yolo26
cd skill/scripts
conda run -n yolo26 python object_detect.py
```

## ManiSkill 资产

ManiSkill 会从以下位置查找外部资产：

```text
$MS_ASSET_DIR/data
```

未设置 `MS_ASSET_DIR` 时，默认位置通常是：

```text
~/.maniskill/data
```

如果使用工程内的资产目录，可以在启动前设置：

```bash
export MS_ASSET_DIR="$PWD/assets"
```

注意：ManiSkill 会自动追加 `/data`。因此上面的配置对应实际目录：

```text
./assets/data
```

## 启动控制服务和仿真

查看 OpenClaw 自定义任务：

```bash
PYTHONPATH=/home/lpx/python-project/VigiClaw uv run python -c "import mani_skill.envs; import openclaw_task; from mani_skill.utils.registration import REGISTERED_ENVS; print('\\n'.join(sorted(k for k in REGISTERED_ENVS if k.startswith('Openclaw'))))"
```



仿真服务端启动时会导入 `openclaw_task`，自动注册工程内的自定义任务。

默认服务地址：

```text
http://127.0.0.1:8000
```

在第一个终端启动仿真服务端（`server.py`，同时提供 HTTP API）：

```bash
cd openclaw_control
uv run python server.py
```

`server.py` 在同一个进程内运行 ManiSkill 仿真和 HTTP API，不需要单独启动 HTTP 客户端服务。
在第二个终端运行 `skill/scripts` 下的控制或检测脚本。

仿真服务端支持通过 `--task` 选择要运行的 OpenClaw 任务，默认任务为 `OpenclawMoveApple`：

```bash
python server.py --task OpenclawPickCube
```

查看帮助：

```bash
python server.py --help
```

仿真端的默认配置写在 [server.py](./openclaw_control/server.py) 中：

```python
SimulationServer(
    host="127.0.0.1",
    port=8000,
    episode_nums=1,
)
```

也可以通过命令行设置 episode 轮数：

```bash
python server.py --task OpenclawPickCube --episode_nums 5
```

任务完成 5 个 episode 后，仿真服务端会自动退出。

当前仿真配置为：

- 任务：`OpenclawMoveApple`
- 机器人：`panda`
- 观测：`rgbd`
- 控制模式：`pd_ee_pose`
- 控制频率：`50 Hz`



## 项目结构

```text
VigiClaw/
├── openclaw_control/
│   ├── server.py       # ManiSkill 仿真端，同时提供 HTTP API
│   └── client.py       # HTTP 客户端工具（非必需）
├── openclaw_task/      # ManiSkill 自定义任务
├── skill/
│   ├── SKILL.md        # 机械臂控制 Skill 文档
│   └── scripts/        # 控制、检测和查询脚本
├── my_examples/        # 示例程序
├── pyproject.toml
└── uv.lock
```

## 注意事项

- 只需要启动 `server.py`，再执行 `skill/scripts` 下的控制脚本，不需要单独启动 `client.py`。
- 控制脚本默认连接 `http://127.0.0.1:8000`。
- `server.py` 当前使用 `render_mode="human"`，无图形环境时需要调整为适合的渲染模式。
- 请不要将模型 checkpoint、仿真资产、检测结果图片或包含个人信息的日志提交到 GitHub。
- 发布前建议检查 `.gitignore`，避免提交 `.venv`、缓存文件和本地生成的 `result.csv`。

## 开源协议

项目协议尚未确定。发布到 GitHub 前，请根据你的需要补充 `LICENSE` 文件。
