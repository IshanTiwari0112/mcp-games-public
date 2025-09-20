# Gym MCP  - AI Gaming Platform

A Model Context Protocol (MCP) server that exposes games as tools for AI agents to discover and play.

## 🚀 Quick Start

### Requirements
- Python 3.10+
- MCP-compatible client (Claude Desktop, etc.)

### Installation

Using pip:
```bash
git clone <this-repo>
cd gym-mcp
pip install -e .
```

Or using uv (recommended by Dedalus Labs):
```bash
git clone <this-repo>
cd gym-mcp
uv sync
```

### Run the Server
```bash
# With pip
python main.py

# With uv
uv run main
```

### Test the Server
```bash
python test_fastmcp.py
```

## 🎮 Available Games

- **Tic-Tac-Toe**: Classic 3x3 grid game with X and O

## 🛠️ Available Tools

### Core Game Tools
- `list_games` - Discover available games and active instances
- `start_game` - Initialize a new game with players
- `make_move` - Execute game actions
- `get_game_state` - View current game state
- `reset_game` - Reset game to initial state

### Game-Specific Tools
- `tic_tac_toe_place_mark` - Place X or O on the board

## 📋 Claude Desktop Integration

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gym-mcp": {
      "command": "python",
      "args": ["main.py"],
      "cwd": "/path/to/gym-mcp"
    }
  }
}
```

## 🎯 Example Usage

```bash
# Discover available games
list_games()

# Start a tic-tac-toe game
start_game(game_type="tic-tac-toe", players=["Alice", "Bob"])

# Make moves using generic tool
make_move(game_id="uuid", action_type="place_mark", payload={"row": 1, "col": 1}, player="Alice")

# Or use game-specific tool
tic_tac_toe_place_mark(game_id="uuid", row=0, col=0, player="Bob")

# Check current state
get_game_state(game_id="uuid")
```

## 🏗️ Architecture

### Core Components
- **FastMCP**: Simple, decorator-based MCP server
- **Game Registry**: Manages available game types and active instances
- **Game Interface**: Abstract base class defining game contract
- **Dynamic Tools**: Each game registers custom MCP tools

### Per-Game Tools
Each game can define its own specific tools:
- **Tic-Tac-Toe**: `tic_tac_toe_place_mark(game_id, row, col, player)`
- **Future games**: Chess could have `chess_castle`, `chess_en_passant`, etc.

### Stateless Design
- Games stored in memory per server session
- No authentication required
- Clean separation between game logic and MCP protocol

## 🔧 Adding New Games

1. **Create game class** inheriting from `Game`:
   ```python
   class Chess(Game):
       def start(self, players): ...
       def make_move(self, action): ...
       def get_state(self): ...
       # etc.
   ```

2. **Define game-specific tools**:
   ```python
   @classmethod
   def get_tool_definitions(cls):
       return [GameToolDefinition(
           name="chess_castle",
           description="Perform castling move",
           input_schema={...}
       )]
   ```

3. **Register in the registry**:
   ```python
   registry.register_game_type(Chess)
   ```

4. **Add FastMCP tool handlers** in `server.py`

## 🌟 Future Plans

- **OpenAI Gym Integration**: Wrap Gymnasium environments
- **PettingZoo Support**: Multi-agent games (Chess, Poker, etc.)
- **Real-time Games**: WebSocket support for faster interactions
- **Tournament Mode**: AI vs AI competitions
- **Spectator Tools**: Watch games in progress

## 🏛️ Architecture Benefits

### For AI Researchers
- Standardized game testing environments
- Consistent APIs across all games
- Easy benchmarking and evaluation

### For Developers
- Simple game integration (just implement the interface)
- Automatic MCP tool generation
- Built-in state management

### For AI Agents
- Unified discovery mechanism (`list_games`)
- Consistent interaction patterns
- Game-specific optimization opportunities

---

Built with [FastMCP](https://github.com/jlowin/fastmcp) following [Dedalus Labs MCP Guidelines](https://docs.dedaluslabs.ai/server-guidelines)
