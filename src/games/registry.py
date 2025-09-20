"""Game registry for managing available games and instances"""

from typing import Dict, List, Optional, Type
from uuid import uuid4

from .types import Game, GameResult
from .tic_tac_toe import TicTacToe
from .cartpole import CartPole
from .breakout import AtariBreakout
from .mountain_car import MountainCar
from .blackjack import Blackjack


class GameRegistry:
    def __init__(self):
        self._game_types: Dict[str, Type[Game]] = {}
        self._game_instances: Dict[str, Game] = {}
        self._register_default_games()

    def _register_default_games(self):
        """Register built-in games"""
        # Custom games
        self.register_game_type(TicTacToe)

        # Gymnasium environments
        try:
            self.register_game_type(CartPole)
            self.register_game_type(MountainCar)
            self.register_game_type(AtariBreakout)
            self.register_game_type(Blackjack)
        except ImportError:
            # Gymnasium not available, skip
            pass

    def register_game_type(self, game_class: Type[Game]):
        """Register a new game type"""
        game_info = game_class.get_game_info()
        self._game_types[game_info["type"]] = game_class

    def get_available_games(self) -> List[Dict]:
        """Get list of available game types"""
        games = []
        for game_type, game_class in self._game_types.items():
            games.append(game_class.get_game_info())
        return games

    def get_active_games(self) -> List[Dict]:
        """Get list of active game instances"""
        return [
            {
                "id": game.id,
                "type": game.type,
                "status": game.state.status,
                "players": game.state.players,
                "moves": game.state.moves
            }
            for game in self._game_instances.values()
        ]

    def create_game(self, game_type: str, game_id: Optional[str] = None) -> GameResult:
        """Create a new game instance"""
        if game_type not in self._game_types:
            return GameResult(
                success=False,
                error=f"Unknown game type: {game_type}"
            )

        if game_id is None:
            game_id = str(uuid4())

        if game_id in self._game_instances:
            return GameResult(
                success=False,
                error=f"Game with ID {game_id} already exists"
            )

        game_class = self._game_types[game_type]
        game_instance = game_class(game_id)
        self._game_instances[game_id] = game_instance

        return GameResult(
            success=True,
            new_state=game_instance.get_state(),
            message=f"Created game {game_id} of type {game_type}"
        )

    def get_game(self, game_id: str) -> Optional[Game]:
        """Get a game instance by ID"""
        return self._game_instances.get(game_id)

    def remove_game(self, game_id: str) -> bool:
        """Remove a game instance"""
        if game_id in self._game_instances:
            del self._game_instances[game_id]
            return True
        return False

    def get_game_tool_definitions(self) -> List[Dict]:
        """Get all game-specific tool definitions"""
        tools = []
        for game_class in self._game_types.values():
            for tool_def in game_class.get_tool_definitions():
                tools.append(tool_def.model_dump())
        return tools