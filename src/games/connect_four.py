"""PettingZoo Connect Four game implementation"""

from typing import Any, Dict, List
from .pettingzoo_game import PettingZooGame
from .types import GameToolDefinition, GameResult, GameAction


class ConnectFour(PettingZooGame):
    """Connect Four game using PettingZoo's connect_four_v3 environment"""

    def __init__(self, game_id: str):
        super().__init__(game_id, "connect_four_v3")

    def _get_text_description(self) -> str:
        """Get human-readable description of current Connect Four state"""
        if not self.state.state:
            return "Game not started"
            
        current_player = self.state.current_player
        episode_steps = self.state.state.get("episode_steps", 0)
        
        description = f"🔴🟡 Connect Four Status:\n"
        description += f"📋 Current turn: {current_player}\n"
        description += f"🎯 Move #{episode_steps + 1}\n"
        
        # Show rewards
        rewards = self.state.state.get("episode_rewards", {})
        if rewards:
            description += f"🏆 Scores: {', '.join([f'{p}: {r}' for p, r in rewards.items()])}\n"
        
        # Check game status
        terminations = self.state.state.get("terminations", {})
        if any(terminations.values()):
            description += f"🏁 Game ended!\n"
            if self.state.winner:
                description += f"🎉 Winner: {self.state.winner}"
            else:
                description += f"🤝 Draw!"
        else:
            description += f"💡 Drop a piece in columns 0-6 using connect4_drop()\n"
            description += f"🎮 Goal: Get 4 pieces in a row (horizontal, vertical, or diagonal)"
            
        return description

    def start(self, players: List[str]) -> GameResult:
        """Start Connect Four game with enhanced descriptions"""
        result = super().start(players)
        if result.success:
            result.new_state.metadata["text_description"] = self._get_text_description()
            result.new_state.metadata["game_type"] = "PettingZoo Connect Four"
        return result

    def make_move(self, action: GameAction) -> GameResult:
        """Make a Connect Four move with enhanced feedback"""
        result = super().make_move(action)
        if result.success:
            result.new_state.metadata["text_description"] = self._get_text_description()
            
            # Add move-specific feedback
            action_value = action.payload.get("action", -1)
            result.new_state.metadata["move_result"] = f"Dropped piece in column {action_value}"
            
            # Add SVG rendering
            svg_board = self._render_game_as_svg()
            if svg_board:
                result.new_state.metadata["svg_render"] = svg_board
                
        return result

    @classmethod
    def get_tool_definitions(cls) -> List[GameToolDefinition]:
        """Connect Four-specific tools"""
        base_tools = super().get_tool_definitions()
        connect4_tools = [
            GameToolDefinition(
                name="connect4_drop",
                description="Drop a piece in Connect Four (column 0-6)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"},
                        "column": {"type": "integer", "minimum": 0, "maximum": 6, "description": "Column to drop piece (0-6)"},
                        "player": {"type": "string", "description": "Player making the move"}
                    },
                    "required": ["game_id", "column", "player"]
                }
            ),
            GameToolDefinition(
                name="connect4_status",
                description="Get current Connect Four game status with board visualization",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"}
                    },
                    "required": ["game_id"]
                }
            )
        ]
        return base_tools + connect4_tools

    @classmethod
    def get_game_info(cls) -> Dict[str, Any]:
        return {
            "type": "connect_four_v3",
            "name": "Connect Four",
            "description": "Classic Connect Four game using PettingZoo. Drop pieces to get 4 in a row!",
            "min_players": 2,
            "max_players": 2,
            "custom_tools": ["connect4_drop", "connect4_status", "pz_step"],
            "action_space": "Discrete(7) - Choose column 0-6 to drop piece",
            "observation_space": "Box - 6x7 grid representing the game board",
            "rules": [
                "Players alternate dropping pieces in columns",
                "Pieces fall to the lowest available position in the column",
                "Win by getting 4 pieces in a row (horizontal, vertical, diagonal)",
                "Game ends in draw if board is full with no winner"
            ],
            "strategy_tips": [
                "Control the center columns for more winning opportunities",
                "Block opponent's potential 4-in-a-row",
                "Look for opportunities to create multiple threats",
                "Watch for diagonal winning patterns"
            ],
            "llm_advantages": [
                "Strategic pattern recognition",
                "Planning multiple moves ahead",
                "Blocking opponent threats",
                "Creating winning combinations"
            ]
        }
