"""Core game types and interfaces"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel


class GameStatus(str, Enum):
    WAITING = "waiting"
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"


class GameState(BaseModel):
    id: str
    type: str
    state: Any
    status: GameStatus
    players: List[str]
    current_player: Optional[str] = None
    winner: Optional[str] = None
    moves: int = 0
    metadata: Dict[str, Any] = {}


class GameAction(BaseModel):
    type: str
    payload: Any
    player: Optional[str] = None


class GameResult(BaseModel):
    success: bool
    new_state: Optional[GameState] = None
    message: Optional[str] = None
    error: Optional[str] = None


class GameToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]


class Game(ABC):
    def __init__(self, game_id: str, game_type: str):
        self.id = game_id
        self.type = game_type
        self.state = GameState(
            id=game_id,
            type=game_type,
            state={},
            status=GameStatus.WAITING,
            players=[],
        )

    @abstractmethod
    def start(self, players: List[str]) -> GameResult:
        """Initialize the game with players"""
        pass

    @abstractmethod
    def make_move(self, action: GameAction) -> GameResult:
        """Process a game action"""
        pass

    @abstractmethod
    def get_state(self) -> GameState:
        """Get current game state"""
        pass

    @abstractmethod
    def reset(self) -> GameResult:
        """Reset the game"""
        pass

    @abstractmethod
    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Get valid actions for current state"""
        pass

    @classmethod
    @abstractmethod
    def get_tool_definitions(cls) -> List[GameToolDefinition]:
        """Get game-specific tool definitions"""
        pass

    @classmethod
    @abstractmethod
    def get_game_info(cls) -> Dict[str, Any]:
        """Get game metadata"""
        pass