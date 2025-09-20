"""Atari Breakout Gymnasium environment"""

from typing import Any, Dict, List
from .gymnasium_game import GymnasiumGame
from .types import GameToolDefinition


class AtariBreakout(GymnasiumGame):
    """ALE/Breakout-v5 environment - classic Atari Breakout game"""

    def __init__(self, game_id: str):
        super().__init__(game_id, "ALE/Breakout-v5")

    @classmethod
    def get_tool_definitions(cls) -> List[GameToolDefinition]:
        """Breakout-specific tools"""
        base_tools = super().get_tool_definitions()
        breakout_tools = [
            GameToolDefinition(
                name="breakout_noop",
                description="Do nothing (action 0)",
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
                name="breakout_fire",
                description="Fire/start the ball (action 1)",
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
                name="breakout_right",
                description="Move paddle right (action 2)",
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
                name="breakout_left",
                description="Move paddle left (action 3)",
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
        return base_tools + breakout_tools

    @classmethod
    def get_game_info(cls) -> Dict[str, Any]:
        return {
            "type": "ALE/Breakout-v5",
            "name": "Atari Breakout",
            "description": "Classic Atari Breakout - destroy bricks with a ball using a paddle. Observation is 210x160x3 RGB image.",
            "min_players": 1,
            "max_players": 1,
            "custom_tools": ["breakout_noop", "breakout_fire", "breakout_right", "breakout_left", "gym_step", "gym_reset"],
            "action_space": "Discrete(4) - 0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT",
            "observation_space": "Box(210, 160, 3) - RGB image",
            "max_episode_steps": 10000,
            "reward_threshold": None
        }