"""FrozenLake game - Navigate a slippery grid to reach the goal"""

from typing import Any, Dict, List
from .gymnasium_game import GymnasiumGame
from .types import GameToolDefinition, GameResult, GameAction


class FrozenLake(GymnasiumGame):
    """FrozenLake-v1 environment - Navigate slippery grid to reach goal"""

    def __init__(self, game_id: str, map_name: str = "4x4"):
        # FrozenLake-v1 supports different map sizes
        env_name = "FrozenLake-v1"
        super().__init__(game_id, env_name)
        self.map_name = map_name
        self.grid_size = 4 if map_name == "4x4" else 8
        
    def _create_env(self):
        """Create FrozenLake environment with specific map"""
        if self.env is None:
            import gymnasium as gym
            self.env = gym.make(
                "FrozenLake-v1", 
                map_name=self.map_name,
                is_slippery=True,  # Makes it more challenging and strategic
                render_mode='ansi'  # Text-based rendering
            )
    
    def _get_position_from_obs(self, obs):
        """Convert flat observation to (row, col) position"""
        return divmod(obs, self.grid_size)
    
    def _get_text_description(self, obs):
        """Get human-readable description of current state"""
        if obs is None:
            return "Game not started"
            
        row, col = self._get_position_from_obs(obs)
        goal_row, goal_col = self.grid_size - 1, self.grid_size - 1
        
        description = f"🧊 You are at position ({row}, {col}) on a {self.grid_size}x{self.grid_size} frozen lake.\n"
        description += f"🎯 Goal: Reach the frisbee at ({goal_row}, {goal_col})\n"
        description += f"⚠️  Warning: Ice is slippery - you might not go exactly where intended!\n"
        description += f"💀 Avoid falling into holes (marked 'H')\n"
        
        # Calculate Manhattan distance to goal
        distance = abs(goal_row - row) + abs(goal_col - col)
        description += f"📏 Distance to goal: {distance} steps"
        
        return description

    def start(self, players: List[str]) -> GameResult:
        """Start FrozenLake game with enhanced text description"""
        result = super().start(players)
        if result.success and self.observation is not None:
            # Add human-readable state description
            result.new_state.metadata["text_description"] = self._get_text_description(self.observation)
            result.new_state.metadata["position"] = self._get_position_from_obs(self.observation)
            result.new_state.metadata["available_moves"] = ["Up", "Down", "Left", "Right"]
        return result

    def make_move(self, action: GameAction) -> GameResult:
        """Make a move with enhanced feedback"""
        result = super().make_move(action)
        if result.success and self.observation is not None:
            # Add descriptive feedback
            result.new_state.metadata["text_description"] = self._get_text_description(self.observation)
            result.new_state.metadata["position"] = self._get_position_from_obs(self.observation)
            
            # Check if game ended
            if self.terminated:
                if result.new_state.state.get("last_reward", 0) > 0:
                    result.new_state.metadata["result"] = "🎉 Success! You reached the frisbee!"
                    result.new_state.winner = action.player
                else:
                    result.new_state.metadata["result"] = "💀 Game Over! You fell into a hole."
        return result

    @classmethod
    def get_tool_definitions(cls) -> List[GameToolDefinition]:
        """FrozenLake-specific tools"""
        base_tools = super().get_tool_definitions()
        frozen_lake_tools = [
            GameToolDefinition(
                name="frozen_lake_move_up",
                description="Move up on the frozen lake grid",
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
                name="frozen_lake_move_down", 
                description="Move down on the frozen lake grid",
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
                name="frozen_lake_move_left",
                description="Move left on the frozen lake grid", 
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
                name="frozen_lake_move_right",
                description="Move right on the frozen lake grid",
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
                name="frozen_lake_status",
                description="Get current FrozenLake game status with map visualization",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"}
                    },
                    "required": ["game_id"]
                }
            )
        ]
        return base_tools + frozen_lake_tools

    @classmethod
    def get_game_info(cls) -> Dict[str, Any]:
        return {
            "type": "FrozenLake-v1",
            "name": "Frozen Lake",
            "description": "Navigate a slippery frozen lake to reach the goal while avoiding holes. Perfect for strategic planning!",
            "min_players": 1,
            "max_players": 1,
            "custom_tools": [
                "frozen_lake_move_up", "frozen_lake_move_down", 
                "frozen_lake_move_left", "frozen_lake_move_right",
                "frozen_lake_status", "gym_step", "gym_reset"
            ],
            "action_space": "Discrete(4) - 0: Left, 1: Down, 2: Right, 3: Up",
            "observation_space": "Discrete(16) - Position on 4x4 grid",
            "rules": [
                "Navigate from start (top-left) to goal (bottom-right)",
                "Ice is slippery - you might not move as intended",
                "Avoid holes (H) - falling in ends the game",
                "Reach the goal (G) to win and get +1 reward"
            ],
            "strategy_tips": [
                "Plan multiple steps ahead due to slippery ice",
                "Consider safer longer paths vs risky shorter paths",
                "Remember: intended action might not be executed due to slipping"
            ]
        }
