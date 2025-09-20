"""Blackjack game - Player vs Dealer"""

from typing import Any, Dict, List
from .gymnasium_game import GymnasiumGame
from .types import GameToolDefinition, GameResult, GameAction


class Blackjack(GymnasiumGame):
    """Blackjack-v1 environment - Classic card game vs dealer"""

    def __init__(self, game_id: str):
        super().__init__(game_id, "Blackjack-v1")
        self.player_sum = 0
        self.dealer_card = 0
        self.usable_ace = False

    def _parse_observation(self, obs):
        """Parse Blackjack observation tuple"""
        if obs:
            self.player_sum = int(obs[0])
            self.dealer_card = int(obs[1])
            self.usable_ace = bool(obs[2])
        return obs

    def start(self, players: List[str]) -> GameResult:
        """Start Blackjack game"""
        result = super().start(players)
        if result.success and self.observation:
            self._parse_observation(self.observation)
            # Add human-readable state
            result.new_state.metadata["player_hand"] = self.player_sum
            result.new_state.metadata["dealer_showing"] = self.dealer_card
            result.new_state.metadata["has_usable_ace"] = self.usable_ace
        return result

    def make_move(self, action: GameAction) -> GameResult:
        """Make a move in Blackjack"""
        result = super().make_move(action)
        if result.success and self.observation:
            self._parse_observation(self.observation)
            # Update metadata with current hand
            result.new_state.metadata["player_hand"] = self.player_sum
            result.new_state.metadata["dealer_showing"] = self.dealer_card
            result.new_state.metadata["has_usable_ace"] = self.usable_ace

            # Check if game ended
            if self.terminated:
                last_reward = result.new_state.state.get("last_reward", 0)
                if last_reward > 0:
                    result.new_state.metadata["result"] = "Player Wins! 🎉"
                    result.new_state.winner = action.player
                elif last_reward < 0:
                    result.new_state.metadata["result"] = "Dealer Wins 😔"
                    result.new_state.winner = "Dealer"
                else:
                    result.new_state.metadata["result"] = "Push (Tie) 🤝"
        return result

    @classmethod
    def get_tool_definitions(cls) -> List[GameToolDefinition]:
        """Blackjack-specific tools"""
        base_tools = super().get_tool_definitions()
        blackjack_tools = [
            GameToolDefinition(
                name="blackjack_hit",
                description="Take another card (Hit)",
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
                name="blackjack_stand",
                description="Keep current hand (Stand)",
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
                name="blackjack_status",
                description="Get current Blackjack game status in readable format",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"}
                    },
                    "required": ["game_id"]
                }
            )
        ]
        return base_tools + blackjack_tools

    @classmethod
    def get_game_info(cls) -> Dict[str, Any]:
        return {
            "type": "Blackjack-v1",
            "name": "Blackjack",
            "description": "Classic Blackjack card game - Player vs Dealer. Try to get 21 without going over!",
            "min_players": 1,
            "max_players": 1,
            "custom_tools": ["blackjack_hit", "blackjack_stand", "blackjack_status", "gym_step", "gym_reset"],
            "action_space": "Discrete(2) - 0: Stand, 1: Hit",
            "observation_space": "Tuple(player_sum[1-31], dealer_card[1-11], usable_ace[0-1])",
            "rules": [
                "Get as close to 21 as possible without going over",
                "Dealer must hit on 16 and stand on 17",
                "Aces can be 1 or 11 (usable ace)",
                "Win: +1 reward, Lose: -1 reward, Tie: 0 reward"
            ]
        }