"""Tic Tac Toe game implementation"""

from typing import Any, Dict, List, Optional
from .types import Game, GameAction, GameResult, GameState, GameStatus, GameToolDefinition


class TicTacToe(Game):
    def __init__(self, game_id: str):
        super().__init__(game_id, "tic-tac-toe")
        self.board = [[None for _ in range(3)] for _ in range(3)]

    def start(self, players: List[str]) -> GameResult:
        if len(players) != 2:
            return GameResult(
                success=False,
                error="Tic-tac-toe requires exactly 2 players"
            )

        self.state.players = players
        self.state.current_player = players[0]
        self.state.status = GameStatus.ACTIVE
        self.state.state = {
            "board": self.board,
            "symbols": {players[0]: "X", players[1]: "O"}
        }

        return GameResult(success=True, new_state=self.state)

    def make_move(self, action: GameAction) -> GameResult:
        if self.state.status != GameStatus.ACTIVE:
            return GameResult(success=False, error="Game is not active")

        if action.player != self.state.current_player:
            return GameResult(success=False, error="Not your turn")

        if action.type != "place_mark":
            return GameResult(success=False, error="Invalid action type")

        row = action.payload.get("row")
        col = action.payload.get("col")

        if not self._is_valid_position(row, col):
            return GameResult(success=False, error="Invalid position")

        if self.board[row][col] is not None:
            return GameResult(success=False, error="Position already taken")

        # Make the move
        symbol = self.state.state["symbols"][action.player]
        self.board[row][col] = symbol
        self.state.moves += 1
        self.state.state["board"] = self.board

        # Check for winner
        winner = self._check_winner()
        if winner:
            self.state.winner = winner
            self.state.status = GameStatus.COMPLETED
        elif self.state.moves >= 9:
            self.state.status = GameStatus.COMPLETED
        else:
            # Switch players
            current_idx = self.state.players.index(self.state.current_player)
            next_idx = (current_idx + 1) % 2
            self.state.current_player = self.state.players[next_idx]

        return GameResult(success=True, new_state=self.state)

    def get_state(self) -> GameState:
        return self.state

    def reset(self) -> GameResult:
        self.board = [[None for _ in range(3)] for _ in range(3)]
        self.state.state = {"board": self.board}
        self.state.status = GameStatus.WAITING
        self.state.current_player = None
        self.state.winner = None
        self.state.moves = 0
        return GameResult(success=True, new_state=self.state)

    def get_available_actions(self) -> List[Dict[str, Any]]:
        if self.state.status != GameStatus.ACTIVE:
            return []

        actions = []
        for row in range(3):
            for col in range(3):
                if self.board[row][col] is None:
                    actions.append({
                        "type": "place_mark",
                        "payload": {"row": row, "col": col},
                        "description": f"Place mark at ({row}, {col})"
                    })
        return actions

    def _is_valid_position(self, row: int, col: int) -> bool:
        return 0 <= row < 3 and 0 <= col < 3

    def _check_winner(self) -> Optional[str]:
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] and row[0] is not None:
                symbol = row[0]
                for player, player_symbol in self.state.state["symbols"].items():
                    if player_symbol == symbol:
                        return player

        # Check columns
        for col in range(3):
            if (self.board[0][col] == self.board[1][col] == self.board[2][col]
                and self.board[0][col] is not None):
                symbol = self.board[0][col]
                for player, player_symbol in self.state.state["symbols"].items():
                    if player_symbol == symbol:
                        return player

        # Check diagonals
        if (self.board[0][0] == self.board[1][1] == self.board[2][2]
            and self.board[0][0] is not None):
            symbol = self.board[0][0]
            for player, player_symbol in self.state.state["symbols"].items():
                if player_symbol == symbol:
                    return player

        if (self.board[0][2] == self.board[1][1] == self.board[2][0]
            and self.board[0][2] is not None):
            symbol = self.board[0][2]
            for player, player_symbol in self.state.state["symbols"].items():
                if player_symbol == symbol:
                    return player

        return None

    @classmethod
    def get_tool_definitions(cls) -> List[GameToolDefinition]:
        return [
            GameToolDefinition(
                name="tic_tac_toe_place_mark",
                description="Place a mark (X or O) on the tic-tac-toe board",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"},
                        "row": {"type": "integer", "minimum": 0, "maximum": 2},
                        "col": {"type": "integer", "minimum": 0, "maximum": 2},
                        "player": {"type": "string", "description": "Player making the move"}
                    },
                    "required": ["game_id", "row", "col", "player"]
                }
            )
        ]

    @classmethod
    def get_game_info(cls) -> Dict[str, Any]:
        return {
            "type": "tic-tac-toe",
            "name": "Tic Tac Toe",
            "description": "Classic 3x3 grid game where players take turns placing X and O",
            "min_players": 2,
            "max_players": 2,
            "custom_tools": ["tic_tac_toe_place_mark"]
        }