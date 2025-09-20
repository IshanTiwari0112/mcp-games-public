"""CartPole-v1 Gymnasium environment"""

from typing import Any, Dict, List
from .gymnasium_game import GymnasiumGame
from .types import GameToolDefinition


class CartPole(GymnasiumGame):
    """CartPole-v1 environment - balance a pole on a cart"""

    def __init__(self, game_id: str):
        super().__init__(game_id, "CartPole-v1")

    @classmethod
    def get_tool_definitions(cls) -> List[GameToolDefinition]:
        """CartPole-specific tools"""
        base_tools = super().get_tool_definitions()
        cartpole_tools = [
            GameToolDefinition(
                name="cartpole_move_left",
                description="Move the cart left (action 0)",
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
                name="cartpole_move_right",
                description="Move the cart right (action 1)",
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
        return base_tools + cartpole_tools

    @classmethod
    def get_game_info(cls) -> Dict[str, Any]:
        return {
            "type": "CartPole-v1",
            "name": "CartPole-v1",
            "description": "Balance a pole on a cart by moving left or right. The observation is [cart_position, cart_velocity, pole_angle, pole_angular_velocity].",
            "min_players": 1,
            "max_players": 1,
            "custom_tools": ["cartpole_move_left", "cartpole_move_right", "gym_step", "gym_reset"],
            "action_space": "Discrete(2) - 0: left, 1: right",
            "observation_space": "Box(4,) - [cart_pos, cart_vel, pole_angle, pole_vel]",
            "max_episode_steps": 500,
            "reward_threshold": 475.0
        }