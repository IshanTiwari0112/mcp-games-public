"""MountainCar-v0 Gymnasium environment"""

from typing import Any, Dict, List
from .gymnasium_game import GymnasiumGame
from .types import GameToolDefinition


class MountainCar(GymnasiumGame):
    """MountainCar-v0 environment - drive up a mountain"""

    def __init__(self, game_id: str):
        super().__init__(game_id, "MountainCar-v0")

    @classmethod
    def get_tool_definitions(cls) -> List[GameToolDefinition]:
        """MountainCar-specific tools"""
        base_tools = super().get_tool_definitions()
        mountain_car_tools = [
            GameToolDefinition(
                name="mountain_car_push_left",
                description="Push car left (action 0)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"},
                        "player": {"type": "string", "description": "Player making the move"}
                    },
                    "required": ["game_id", "player"]
                }
            ),
            GameToolDefinition(
                name="mountain_car_no_push",
                description="Don't push (action 1)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"},
                        "player": {"type": "string", "description": "Player making the move"}
                    },
                    "required": ["game_id", "player"]
                }
            ),
            GameToolDefinition(
                name="mountain_car_push_right",
                description="Push car right (action 2)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"},
                        "player": {"type": "string", "description": "Player making the move"}
                    },
                    "required": ["game_id", "player"]
                }
            )
        ]
        return base_tools + mountain_car_tools

    @classmethod
    def get_game_info(cls) -> Dict[str, Any]:
        return {
            "type": "MountainCar-v0",
            "name": "MountainCar-v0",
            "description": "Drive a car up a mountain by building momentum. Observation is [position, velocity].",
            "min_players": 1,
            "max_players": 1,
            "custom_tools": ["mountain_car_push_left", "mountain_car_no_push", "mountain_car_push_right", "gym_step", "gym_reset"],
            "action_space": "Discrete(3) - 0: left, 1: nothing, 2: right",
            "observation_space": "Box(2,) - [position, velocity]",
            "max_episode_steps": 200,
            "reward_threshold": -110.0
        }