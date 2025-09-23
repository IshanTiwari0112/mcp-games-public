"""PettingZoo Chess game implementation"""

from typing import Any, Dict, List
from .pettingzoo_game import PettingZooGame
from .types import GameToolDefinition, GameResult, GameAction


class PettingZooChess(PettingZooGame):
    """Chess game using PettingZoo's chess_v6 environment"""

    def __init__(self, game_id: str):
        super().__init__(game_id, "chess_v6")

    def _get_text_description(self) -> str:
        """Get human-readable description of current chess state"""
        if not self.state.state:
            return "Game not started"
            
        current_player = self.state.current_player
        episode_steps = self.state.state.get("episode_steps", 0)
        
        description = f"♟️ PettingZoo Chess Status:\n"
        description += f"📋 Current turn: {current_player}\n"
        description += f"🎯 Move #{episode_steps + 1}\n"
        
        # Get current agent's observation
        observations = self.state.state.get("observations", {})
        if current_player in observations:
            obs = observations[current_player]
            if isinstance(obs, dict) and obs.get("type") == "array":
                # Chess observation is typically a large array representing the board
                description += f"📊 Board state: {obs['shape']} array (chess position encoding)\n"
        
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
            description += f"💡 Use chess_move_pz() to make moves\n"
            
        return description

    def start(self, players: List[str]) -> GameResult:
        """Start PettingZoo chess game with enhanced descriptions"""
        result = super().start(players)
        if result.success:
            result.new_state.metadata["text_description"] = self._get_text_description()
            result.new_state.metadata["game_type"] = "PettingZoo Chess"
        return result

    def make_move(self, action: GameAction) -> GameResult:
        """Make a chess move with enhanced feedback"""
        result = super().make_move(action)
        if result.success:
            result.new_state.metadata["text_description"] = self._get_text_description()
            
            # Add SVG rendering
            svg_render = self._render_game_as_svg()
            if svg_render:
                result.new_state.metadata["svg_render"] = svg_render
            
            # Add move-specific feedback
            last_rewards = result.new_state.state.get("last_rewards", {})
            if action.player in last_rewards:
                reward = last_rewards[action.player]
                if reward > 0:
                    result.new_state.metadata["move_result"] = f"Great move! +{reward} points"
                elif reward < 0:
                    result.new_state.metadata["move_result"] = f"Move penalty: {reward} points"
                else:
                    result.new_state.metadata["move_result"] = "Move completed"
                    
        return result

    @classmethod
    def get_tool_definitions(cls) -> List[GameToolDefinition]:
        """PettingZoo Chess-specific tools"""
        base_tools = super().get_tool_definitions()
        chess_tools = [
            GameToolDefinition(
                name="chess_move_pz",
                description="Make a chess move in PettingZoo chess (action number 0-4095)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"},
                        "action": {"type": "integer", "minimum": 0, "maximum": 4095, "description": "Chess move action (0-4095)"},
                        "player": {"type": "string", "description": "Player making the move"}
                    },
                    "required": ["game_id", "action", "player"]
                }
            ),
            GameToolDefinition(
                name="chess_status_pz",
                description="Get current PettingZoo chess game status with detailed information",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"}
                    },
                    "required": ["game_id"]
                }
            )
        ]
        return base_tools + chess_tools

    @classmethod
    def get_game_info(cls) -> Dict[str, Any]:
        return {
            "type": "chess_v6",
            "name": "PettingZoo Chess",
            "description": "Chess game using PettingZoo's battle-tested chess_v6 environment. Multi-agent chess with full rules!",
            "min_players": 2,
            "max_players": 2,
            "custom_tools": ["chess_move_pz", "chess_status_pz", "pz_step"],
            "action_space": "Discrete(4096) - All possible chess moves encoded as integers",
            "observation_space": "Box - Chess board state as multi-dimensional array",
            "rules": [
                "Standard chess rules using PettingZoo implementation",
                "White moves first (player_0), Black second (player_1)",
                "Actions are encoded as integers 0-4095 representing all possible moves",
                "Game ends on checkmate, stalemate, or draw conditions"
            ],
            "strategy_tips": [
                "PettingZoo chess uses integer action encoding (0-4095)",
                "Observation is a complex array representing full board state",
                "Rewards: +1 for win, -1 for loss, 0 for draw",
                "Use chess_move_pz() for move-specific interface"
            ],
            "llm_advantages": [
                "Battle-tested PettingZoo implementation",
                "Full chess rules with proper multi-agent handling", 
                "Standardized action/observation spaces",
                "Reliable termination and reward signals"
            ]
        }
