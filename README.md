# MCP Games Server

A Model Context Protocol (MCP) server that provides AI agents with access to a variety of classic games, Gymnasium environments, and board games through a unified tool interface.

## Features

- **33+ MCP Tools** for game interaction
- **Visual Rendering** - Returns game states as base64 PNG images or SVG
- **Multiple Game Types** - Classic control, board games, Atari games
- **Unified Interface** - Consistent API across all game types
- **Rich Game State** - Visual and textual game state information

## Installation

```bash
# Clone the repository
git clone https://github.com/itsish/mcp-games-public
cd mcp-games-public

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m src.main
```

## Quick Start

### For Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mcp-games": {
      "command": "python",
      "args": ["-m", "src.main"],
      "cwd": "/path/to/mcp-games-public"
    }
  }
}
```

### For MCP Inspector

```bash
npx @modelcontextprotocol/inspector python -m src.main
```

## Supported Games

### Classic Control (Gymnasium)
- **CartPole-v1** - Balance a pole on a moving cart
- **MountainCar-v0** - Drive a car up a steep mountain
- **FrozenLake-v1** - Navigate a frozen lake avoiding holes

### Atari Games
- **Breakout** - Classic brick-breaking arcade game

### Board Games
- **Tic-Tac-Toe** - Classic 3x3 grid game
- **Chess** - Complete chess implementation with legal move validation
- **Connect Four** - Drop pieces to connect four in a row
- **Blackjack** - Casino card game (hit or stand to reach 21)

### Multi-Agent Games (PettingZoo)
- **Chess (PettingZoo)** - Two-player chess with full rule implementation

## Available Tools

### Game Management
- `list_games` - List all available games
- `start_game` - Start a new game instance
- `get_game_state` - Get current game state with visual rendering
- `reset_game` - Reset game to initial state
- `end_game` - End and cleanup game instance

### CartPole Actions
- `cartpole_move_left` - Move cart left
- `cartpole_move_right` - Move cart right
- `cartpole_get_state` - Get cart position, velocity, pole angle

### MountainCar Actions
- `mountaincar_accelerate_left` - Accelerate car left
- `mountaincar_accelerate_right` - Accelerate car right
- `mountaincar_coast` - Coast without acceleration

### Breakout Actions
- `breakout_start` - Start new Breakout game
- `breakout_move` - Move paddle (left/right/stay)
- `breakout_get_state` - Get game state with visual frame

### Blackjack Actions
- `blackjack_start` - Start new Blackjack game
- `blackjack_hit` - Draw another card
- `blackjack_stand` - Keep current hand
- `blackjack_get_state` - Get current hand and dealer's visible card

### Tic-Tac-Toe Actions
- `tictactoe_start` - Start new Tic-Tac-Toe game
- `tictactoe_place` - Place X or O at position
- `tictactoe_get_state` - Get board state

### FrozenLake Actions
- `frozenlake_move_up` - Move player up
- `frozenlake_move_down` - Move player down
- `frozenlake_move_left` - Move player left
- `frozenlake_move_right` - Move player right

### Chess Actions
- `chess_move` - Make a chess move (e.g., "e2e4")
- `chess_get_legal_moves` - Get all legal moves for current position
- `chess_analyze_position` - Get position analysis and evaluation

### Connect Four Actions
- `connectfour_start` - Start new Connect Four game
- `connectfour_drop_piece` - Drop piece in column (0-6)
- `connectfour_get_state` - Get board state with visual
- `connectfour_get_valid_moves` - Get list of valid columns
- `connectfour_undo_move` - Undo last move
- `connectfour_get_winner` - Check if game has a winner

### PettingZoo Chess Actions
- `pettingzoo_chess_start` - Start PettingZoo chess game
- `pettingzoo_chess_move` - Make move using action index
- `pettingzoo_chess_get_state` - Get board observation
- `pettingzoo_chess_get_legal_actions` - Get legal action indices

## Example Usage

```python
# Start a game
response = await start_game(game_type="CartPole-v1", player_name="AI_Agent")

# Get visual state
state = await cartpole_get_state(game_id=response.game_id)
# Returns state with base64 PNG image of current frame

# Make a move
result = await cartpole_move_right(game_id=response.game_id)

# For chess
await chess_move(game_id=chess_id, move="e2e4")
legal_moves = await chess_get_legal_moves(game_id=chess_id)
```

## Visual Rendering

All games support visual rendering in responses:
- **PNG Images**: Returned as base64-encoded strings for pixel-based games
- **SVG Rendering**: Vector graphics for board games and simple environments
- **Automatic Format Selection**: Games choose the best visualization format

## Architecture

- **FastMCP Framework**: Simple, decorator-based MCP server implementation
- **Game Registry**: Centralized management of game types and instances
- **Gymnasium Integration**: Direct integration with OpenAI Gym environments
- **Modular Design**: Easy to add new games and environments

## Requirements

- Python 3.10+
- See `requirements.txt` for package dependencies

## License

MIT License

## Contributing

Contributions welcome! Feel free to:
- Add new games or environments
- Improve visualization rendering
- Enhance game state representations
- Fix bugs or improve documentation

## Support

For issues or questions, please open an issue on GitHub.