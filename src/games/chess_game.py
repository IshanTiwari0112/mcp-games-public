"""Chess game using python-chess library"""

import chess
import chess.engine
from typing import Any, Dict, List, Optional
from .types import Game, GameAction, GameResult, GameState, GameStatus, GameToolDefinition


class ChessGame(Game):
    """Chess game implementation using python-chess library"""

    def __init__(self, game_id: str):
        super().__init__(game_id, "chess")
        self.board = chess.Board()
        self.move_history = []

    def start(self, players: List[str]) -> GameResult:
        """Start chess game with two players"""
        if len(players) != 2:
            return GameResult(
                success=False,
                error="Chess requires exactly 2 players"
            )

        self.state.players = players
        self.state.current_player = players[0]  # White goes first
        self.state.status = GameStatus.ACTIVE
        self.state.state = {
            "board_fen": self.board.fen(),
            "board_ascii": str(self.board),
            "legal_moves": [move.uci() for move in self.board.legal_moves],
            "turn": "white",
            "move_count": 0,
            "check": False,
            "checkmate": False,
            "stalemate": False
        }

        # Add metadata with game description
        self.state.metadata["game_description"] = self._get_game_description()

        return GameResult(success=True, new_state=self.state)

    def make_move(self, action: GameAction) -> GameResult:
        """Make a chess move"""
        if self.state.status != GameStatus.ACTIVE:
            return GameResult(success=False, error="Game is not active")

        if action.player != self.state.current_player:
            return GameResult(success=False, error="Not your turn")

        if action.type != "chess_move":
            return GameResult(success=False, error="Invalid action type")

        move_uci = action.payload.get("move")
        if not move_uci:
            return GameResult(success=False, error="Move not specified")

        try:
            # Parse and validate the move
            move = chess.Move.from_uci(move_uci)
            if move not in self.board.legal_moves:
                legal_moves_sample = [m.uci() for m in list(self.board.legal_moves)[:10]]
                return GameResult(
                    success=False, 
                    error=f"Illegal move: {move_uci}. Legal moves: {legal_moves_sample}..."
                )

            # Make the move
            self.board.push(move)
            self.move_history.append(move_uci)
            self.state.moves += 1

            # Update game state
            self._update_game_state()

            # Switch players
            current_idx = self.state.players.index(self.state.current_player)
            self.state.current_player = self.state.players[1 - current_idx]

            # Check for game end
            if self.board.is_checkmate():
                self.state.status = GameStatus.COMPLETED
                # Winner is the player who just moved (since opponent is in checkmate)
                winner_idx = 1 - current_idx
                self.state.winner = self.state.players[winner_idx]
                self.state.metadata["result"] = f"Checkmate! {self.state.winner} wins!"
            elif self.board.is_stalemate():
                self.state.status = GameStatus.COMPLETED
                self.state.metadata["result"] = "Stalemate - Draw!"
            elif self.board.is_insufficient_material():
                self.state.status = GameStatus.COMPLETED
                self.state.metadata["result"] = "Draw - Insufficient material!"
            elif self.board.is_fivefold_repetition():
                self.state.status = GameStatus.COMPLETED
                self.state.metadata["result"] = "Draw - Fivefold repetition!"

            return GameResult(success=True, new_state=self.state)

        except ValueError as e:
            return GameResult(success=False, error=f"Invalid move format: {str(e)}")
        except Exception as e:
            return GameResult(success=False, error=f"Move failed: {str(e)}")

    def _update_game_state(self):
        """Update the game state after a move"""
        self.state.state.update({
            "board_fen": self.board.fen(),
            "board_ascii": str(self.board),
            "legal_moves": [move.uci() for move in self.board.legal_moves],
            "turn": "white" if self.board.turn else "black",
            "move_count": len(self.move_history),
            "check": self.board.is_check(),
            "checkmate": self.board.is_checkmate(),
            "stalemate": self.board.is_stalemate(),
            "last_move": self.move_history[-1] if self.move_history else None
        })

        # Add game description
        self.state.metadata["game_description"] = self._get_game_description()

    def _get_game_description(self) -> str:
        """Get human-readable description of current game state"""
        description = f"♟️ Chess Game Status:\n"
        description += f"📋 Current turn: {self.state.state.get('turn', 'white').title()}\n"
        description += f"🎯 Move #{self.state.state.get('move_count', 0) + 1}\n"
        
        if self.state.state.get('check'):
            description += f"⚠️ CHECK! The {self.state.state.get('turn', 'white')} king is in check!\n"
        
        if self.state.state.get('checkmate'):
            description += f"🏆 CHECKMATE! Game Over!\n"
        elif self.state.state.get('stalemate'):
            description += f"🤝 STALEMATE! It's a draw!\n"
        
        last_move = self.state.state.get('last_move')
        if last_move:
            description += f"📝 Last move: {last_move}\n"
        
        # Show a few example legal moves
        legal_moves = self.state.state.get('legal_moves', [])
        if legal_moves:
            example_moves = legal_moves[:5]
            description += f"💡 Example legal moves: {', '.join(example_moves)}"
            if len(legal_moves) > 5:
                description += f" (and {len(legal_moves) - 5} more)"
        
        description += f"\n\n📋 Current Board:\n{self.state.state.get('board_ascii', 'No board')}"
        
        return description

    def get_state(self) -> GameState:
        """Get current game state"""
        return self.state

    def reset(self) -> GameResult:
        """Reset the chess game"""
        self.board = chess.Board()
        self.move_history = []
        self.state.status = GameStatus.WAITING
        self.state.current_player = None
        self.state.winner = None
        self.state.moves = 0
        self.state.state = {
            "board_fen": self.board.fen(),
            "board_ascii": str(self.board),
            "legal_moves": [move.uci() for move in self.board.legal_moves],
            "turn": "white",
            "move_count": 0,
            "check": False,
            "checkmate": False,
            "stalemate": False
        }
        self.state.metadata = {}
        return GameResult(success=True, new_state=self.state)

    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Get available chess moves"""
        if self.state.status != GameStatus.ACTIVE:
            return []

        actions = []
        for move in self.board.legal_moves:
            # Try to get piece and square info
            piece = self.board.piece_at(move.from_square)
            piece_name = piece.symbol().upper() if piece else "?"
            
            actions.append({
                "type": "chess_move",
                "move": move.uci(),
                "description": f"{piece_name} from {chess.square_name(move.from_square)} to {chess.square_name(move.to_square)}",
                "algebraic": self.board.san(move)  # Standard algebraic notation
            })
        
        return actions

    @classmethod
    def get_tool_definitions(cls) -> List[GameToolDefinition]:
        """Chess-specific tool definitions"""
        return [
            GameToolDefinition(
                name="chess_move",
                description="Make a chess move using UCI notation (e.g., 'e2e4', 'g1f3')",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"},
                        "move": {
                            "type": "string", 
                            "description": "Move in UCI notation (e.g., 'e2e4' for pawn from e2 to e4)"
                        },
                        "player": {"type": "string", "description": "Player making the move"}
                    },
                    "required": ["game_id", "move", "player"]
                }
            ),
            GameToolDefinition(
                name="chess_get_legal_moves",
                description="Get all legal moves in the current position",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"}
                    },
                    "required": ["game_id"]
                }
            ),
            GameToolDefinition(
                name="chess_get_board",
                description="Get current board position with detailed analysis",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"}
                    },
                    "required": ["game_id"]
                }
            ),
            GameToolDefinition(
                name="chess_analyze_position",
                description="Get strategic analysis of the current chess position",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"}
                    },
                    "required": ["game_id"]
                }
            )
        ]

    @classmethod
    def get_game_info(cls) -> Dict[str, Any]:
        return {
            "type": "chess",
            "name": "Chess",
            "description": "Classic chess game with full rules implementation. Perfect for strategic thinking and planning!",
            "min_players": 2,
            "max_players": 2,
            "custom_tools": [
                "chess_move", "chess_get_legal_moves", "chess_get_board", "chess_analyze_position"
            ],
            "rules": [
                "Standard chess rules apply",
                "White moves first",
                "Players alternate turns",
                "Win by checkmate, lose by being checkmated",
                "Draw by stalemate, insufficient material, or repetition"
            ],
            "strategy_tips": [
                "Control the center with pawns and pieces",
                "Develop pieces before attacking",
                "Keep your king safe (castle early)",
                "Look for tactical opportunities (forks, pins, skewers)",
                "Think several moves ahead",
                "Consider your opponent's threats and plans"
            ],
            "move_format": "UCI notation (e.g., 'e2e4', 'g1f3', 'e1g1' for castling)",
            "llm_advantages": [
                "Strategic planning and evaluation",
                "Pattern recognition in positions", 
                "Opening knowledge from training data",
                "Tactical calculation and analysis",
                "Endgame technique understanding"
            ]
        }
