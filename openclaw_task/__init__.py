"""OpenClaw tasks built on top of the installed ManiSkill package.

Importing this package registers all OpenClaw environments with ManiSkill and
Gymnasium.  The registration is intentionally kept here instead of relying on
ManiSkill to discover task modules from this project automatically.
"""

from .daily_scene import DailySceneEnv
from .move_apple import MoveAppleEnv
from .obstacle_avoid import ObstacleAvoidEnv
from .open_cabinet import OpenCabinetEnv
from .pick_banana import PickBananaEnv
from .pick_cube import PickCubeEnv
from .pick_cup import PickCupEnv
from .push_cube import PushCubeEnv
from .put_banana_on_plate import PutBananaOnPlateEnv
from .put_together import PutTogetherEnv
from .put_together_disturb import PutTogetherDisturbEnv
from .stack_cubes import StackCubesEnv
from .stack_cubes_disturb import StackCubesDisturbEnv


TASK_ENV_IDS = (
    "OpenclawPickCube",
    "OpenclawPickCup",
    "OpenclawPickBanana",
    "OpenclawMoveApple",
    "OpenclawPutBananaOnPlate",
    "OpenclawStackCubes",
    "OpenclawDailyScene",
    "OpenclawPutTogether",
    "OpenclawStackCubesDisturb",
    "OpenclawPutTogetherDisturb",
    "OpenclawOpenCabinet",
    "OpenclawPushCube",
    "OpenclawObstacleAvoid",
)


def register_tasks() -> tuple[str, ...]:
    """Import OpenClaw task modules and return their registered environment IDs.

    The imports above perform registration through ManiSkill's
    ``@register_env`` decorator.  This function gives callers an explicit,
    discoverable entry point while remaining safe to call repeatedly because
    Python caches imported modules.
    """

    return TASK_ENV_IDS


__all__ = [
    "TASK_ENV_IDS",
    "register_tasks",
    "DailySceneEnv",
    "MoveAppleEnv",
    "ObstacleAvoidEnv",
    "OpenCabinetEnv",
    "PickBananaEnv",
    "PickCubeEnv",
    "PickCupEnv",
    "PushCubeEnv",
    "PutBananaOnPlateEnv",
    "PutTogetherEnv",
    "PutTogetherDisturbEnv",
    "StackCubesEnv",
    "StackCubesDisturbEnv",
]
