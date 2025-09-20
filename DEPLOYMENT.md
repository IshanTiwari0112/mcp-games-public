# Gym MCP Deployment Guide

## Architecture Summary

Our **Gym MCP Server** now supports both transport methods:

### 1. **Local Claude Desktop** (STDIO Transport)
- For development and local Claude Desktop usage
- Uses `transport='stdio'`
- Run: `python main.py --stdio`

### 2. **Dedalus Platform** (Streamable HTTP Transport)
- For production deployment on Dedalus hosted platform
- Uses `transport='streamable-http'`
- Run: `python main.py --port 8080`

## Features

- **20+ Game Tools**: Including Blackjack, CartPole, Breakout, MountainCar, and Tic-Tac-Toe
- **Visual Web Viewer**: Real-time game display at http://localhost:8000
- **Gymnasium Integration**: Full OpenAI Gym and Atari environment support
- **Game Registry**: Dynamic game type management

## Dedalus Deployment

### Requirements
- Python 3.10+
- FastMCP with streamable HTTP support
- Environment variable `PORT` for HTTP deployment

### Deployment Command
```bash
# For Dedalus platform (HTTP transport)
PORT=8080 python main.py

# For local development (STDIO transport)
python main.py --stdio
```

### Environment Variables
- `PORT`: HTTP port for streamable transport (required for Dedalus)
- `DEDALUS_API_KEY`: API key for Dedalus platform (if needed)

## Architecture Compliance

✅ **Dedalus Requirements Met:**
- Main entry point: `main.py` at root
- Source code: `src/main.py`
- Streamable HTTP transport support
- FastMCP framework usage
- Command line argument parsing
- Environment variable configuration

## Game Tools Available

### Core Tools
- `list_games()` - List available game types and active instances
- `start_game(game_type, players)` - Start a new game
- `get_game_state(game_id)` - Get current game state
- `reset_game(game_id)` - Reset a game

### Game-Specific Tools
- **Blackjack**: `blackjack_hit`, `blackjack_stand`, `blackjack_status`
- **CartPole**: `cartpole_move_left`, `cartpole_move_right`
- **Breakout**: `breakout_fire`, `breakout_left`, `breakout_right`, `breakout_noop`
- **MountainCar**: `mountain_car_push_left`, `mountain_car_push_right`, `mountain_car_no_push`
- **Tic-Tac-Toe**: `tic_tac_toe_place_mark`

### Generic Gym Tools
- `gym_step(game_id, action, player)` - Take action in any Gym environment
- `gym_reset(game_id)` - Reset any Gym environment

## Testing

```bash
# Test configuration
python main.py --test

# Test HTTP transport locally
python main.py --port 8001

# Test STDIO transport
python main.py --stdio
```

## Next Steps

1. Deploy to Dedalus platform
2. Configure with Dedalus API
3. Test game functionality through Dedalus gateway
4. Share server URL with AI agents