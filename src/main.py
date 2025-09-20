"""Main MCP server implementation using FastMCP"""

import argparse
import asyncio
import os
import sys
import base64
import io
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

try:
    from .games.registry import GameRegistry
    try:
        from .games.types import GameAction
    except ImportError:
        from games.types import GameAction
except ImportError:
    # Fallback for when module is imported directly
    from games.registry import GameRegistry
    from games.types import GameAction

# Initialize FastMCP server
mcp = FastMCP("gym-mcp")

# Global registry instance
registry = GameRegistry()


def get_game_frame(game_id: str) -> Optional[str]:
    """Get the current visual frame of a game as base64 encoded image"""
    game = registry.get_game(game_id)
    if not game:
        return None

    try:
        # Try to get the environment's rendered frame
        if hasattr(game, 'env') and game.env is not None:
            # First check if we can render
            try:
                frame = game.env.render()
                if frame is not None:
                    import numpy as np
                    from PIL import Image

                    # Convert frame to PIL Image if it's a numpy array
                    if isinstance(frame, np.ndarray):
                        if frame.dtype != np.uint8:
                            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

                        img = Image.fromarray(frame)

                        # Convert to base64
                        buffer = io.BytesIO()
                        img.save(buffer, format='PNG')
                        img_str = base64.b64encode(buffer.getvalue()).decode()
                        return f"data:image/png;base64,{img_str}"
            except Exception as render_error:
                # Try alternative render mode
                try:
                    # Re-create environment with correct render mode if needed
                    import gymnasium as gym
                    import numpy as np
                    from PIL import Image

                    env_name = getattr(game, 'env_name', game.type)

                    # Create a temporary environment with correct render mode
                    temp_env = gym.make(env_name, render_mode='rgb_array')

                    # Reset to initial state
                    temp_env.reset()

                    frame = temp_env.render()
                    temp_env.close()

                    if frame is not None and isinstance(frame, np.ndarray):
                        if frame.dtype != np.uint8:
                            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

                        img = Image.fromarray(frame)
                        buffer = io.BytesIO()
                        img.save(buffer, format='PNG')
                        img_str = base64.b64encode(buffer.getvalue()).decode()
                        return f"data:image/png;base64,{img_str}"
                except Exception:
                    pass  # Silently fail fallback

        # Fallback: try to get frame from observation if it's an image
        state = game.get_state()
        if state and state.state and 'observation' in state.state:
            obs = state.state['observation']
            if isinstance(obs, dict) and obs.get('type') == 'image':
                return f"data:image/png;base64,{obs['data']}"

    except Exception as e:
        print(f"Error getting frame for game {game_id}: {e}", file=sys.stderr)

    return None


def format_response_with_frame(message: str, game_id: str) -> str:
    """Format response to include visual frame if available"""
    frame = get_game_frame(game_id)
    if frame:
        return f"{message}\n\n🎮 Current Frame:\n![Game Frame]({frame})"
    return message


@mcp.tool()
def list_games(show_active: bool = True) -> str:
    """List available game types and active game instances"""
    available_games = registry.get_available_games()
    result = {"available_games": available_games}

    if show_active:
        active_games = registry.get_active_games()
        result["active_games"] = active_games

    return f"Games: {result}"


@mcp.tool()
def start_game(game_type: str, players: List[str], game_id: Optional[str] = None) -> str:
    """Start a new game instance"""
    # Create game
    result = registry.create_game(game_type, game_id)
    if not result.success:
        return f"Error: {result.error}"

    # Start game with players
    game = registry.get_game(result.new_state.id)
    start_result = game.start(players)

    if start_result.success:
        return f"Game started: {start_result.new_state.model_dump()}"
    else:
        return f"Error starting game: {start_result.error}"


@mcp.tool()
def make_move(game_id: str, action_type: str, payload: Dict[str, Any], player: str) -> str:
    """Make a move in an active game"""
    game = registry.get_game(game_id)
    if not game:
        return f"Game {game_id} not found"

    action = GameAction(type=action_type, payload=payload, player=player)
    result = game.make_move(action)

    if result.success:
        base_msg = f"Move successful: {result.new_state.model_dump()}"
        return format_response_with_frame(base_msg, game_id)
    else:
        return f"Move failed: {result.error}"


@mcp.tool()
def get_game_state(game_id: str) -> str:
    """Get current state of a game"""
    game = registry.get_game(game_id)
    if not game:
        return f"Game {game_id} not found"

    state = game.get_state()
    base_msg = f"Game state: {state.model_dump()}"
    return format_response_with_frame(base_msg, game_id)


@mcp.tool()
def reset_game(game_id: str) -> str:
    """Reset a game to initial state"""
    game = registry.get_game(game_id)
    if not game:
        return f"Game {game_id} not found"

    result = game.reset()
    if result.success:
        return f"Game reset: {result.new_state.model_dump()}"
    else:
        return f"Reset failed: {result.error}"


@mcp.tool()
def tic_tac_toe_place_mark(game_id: str, row: int, col: int, player: str) -> str:
    """Place a mark (X or O) on the tic-tac-toe board"""
    game = registry.get_game(game_id)
    if not game or game.type != "tic-tac-toe":
        return f"Tic-tac-toe game {game_id} not found"

    action = GameAction(
        type="place_mark",
        payload={"row": row, "col": col},
        player=player
    )
    result = game.make_move(action)

    if result.success:
        return f"Mark placed: {result.new_state.model_dump()}"
    else:
        return f"Failed to place mark: {result.error}"


# Gymnasium-specific tools
@mcp.tool()
def gym_step(game_id: str, action: Any, player: str) -> str:
    """Take a step in a Gymnasium environment"""
    game = registry.get_game(game_id)
    if not game:
        return f"Game {game_id} not found"

    action_obj = GameAction(type="gym_step", payload={"action": action}, player=player)
    result = game.make_move(action_obj)

    if result.success:
        base_msg = f"Step successful: {result.new_state.model_dump()}"
        return format_response_with_frame(base_msg, game_id)
    else:
        return f"Step failed: {result.error}"


@mcp.tool()
def gym_reset(game_id: str) -> str:
    """Reset a Gymnasium environment"""
    game = registry.get_game(game_id)
    if not game:
        return f"Game {game_id} not found"

    result = game.reset()
    if result.success:
        return f"Environment reset: {result.new_state.model_dump()}"
    else:
        return f"Reset failed: {result.error}"


# CartPole-specific tools
@mcp.tool()
def cartpole_move_left(game_id: str, player: str) -> str:
    """Move CartPole cart left"""
    game = registry.get_game(game_id)
    if not game or game.type != "CartPole-v1":
        return f"CartPole game {game_id} not found"

    action_obj = GameAction(type="gym_step", payload={"action": 0}, player=player)
    result = game.make_move(action_obj)

    if result.success:
        base_msg = f"Moved left: {result.new_state.model_dump()}"
        return format_response_with_frame(base_msg, game_id)
    else:
        return f"Move failed: {result.error}"


@mcp.tool()
def cartpole_move_right(game_id: str, player: str) -> str:
    """Move CartPole cart right"""
    game = registry.get_game(game_id)
    if not game or game.type != "CartPole-v1":
        return f"CartPole game {game_id} not found"

    action_obj = GameAction(type="gym_step", payload={"action": 1}, player=player)
    result = game.make_move(action_obj)

    if result.success:
        base_msg = f"Moved right: {result.new_state.model_dump()}"
        return format_response_with_frame(base_msg, game_id)
    else:
        return f"Move failed: {result.error}"


# MountainCar-specific tools
@mcp.tool()
def mountain_car_push_left(game_id: str, player: str) -> str:
    """Push MountainCar left"""
    game = registry.get_game(game_id)
    if not game or game.type != "MountainCar-v0":
        return f"MountainCar game {game_id} not found"

    action_obj = GameAction(type="gym_step", payload={"action": 0}, player=player)
    result = game.make_move(action_obj)

    if result.success:
        return f"Pushed left: {result.new_state.model_dump()}"
    else:
        return f"Move failed: {result.error}"


@mcp.tool()
def mountain_car_no_push(game_id: str, player: str) -> str:
    """Don't push MountainCar"""
    game = registry.get_game(game_id)
    if not game or game.type != "MountainCar-v0":
        return f"MountainCar game {game_id} not found"

    action_obj = GameAction(type="gym_step", payload={"action": 1}, player=player)
    result = game.make_move(action_obj)

    if result.success:
        return f"No push: {result.new_state.model_dump()}"
    else:
        return f"Move failed: {result.error}"


@mcp.tool()
def mountain_car_push_right(game_id: str, player: str) -> str:
    """Push MountainCar right"""
    game = registry.get_game(game_id)
    if not game or game.type != "MountainCar-v0":
        return f"MountainCar game {game_id} not found"

    action_obj = GameAction(type="gym_step", payload={"action": 2}, player=player)
    result = game.make_move(action_obj)

    if result.success:
        return f"Pushed right: {result.new_state.model_dump()}"
    else:
        return f"Move failed: {result.error}"


# Breakout-specific tools
@mcp.tool()
def breakout_noop(game_id: str, player: str) -> str:
    """Do nothing in Breakout"""
    game = registry.get_game(game_id)
    if not game or game.type != "ALE/Breakout-v5":
        return f"Breakout game {game_id} not found"

    action_obj = GameAction(type="gym_step", payload={"action": 0}, player=player)
    result = game.make_move(action_obj)

    if result.success:
        base_msg = f"No-op: {result.new_state.model_dump()}"
        return format_response_with_frame(base_msg, game_id)
    else:
        return f"Action failed: {result.error}"


@mcp.tool()
def breakout_fire(game_id: str, player: str) -> str:
    """Fire ball in Breakout"""
    game = registry.get_game(game_id)
    if not game or game.type != "ALE/Breakout-v5":
        return f"Breakout game {game_id} not found"

    action_obj = GameAction(type="gym_step", payload={"action": 1}, player=player)
    result = game.make_move(action_obj)

    if result.success:
        base_msg = f"Fired: {result.new_state.model_dump()}"
        return format_response_with_frame(base_msg, game_id)
    else:
        return f"Action failed: {result.error}"


@mcp.tool()
def breakout_right(game_id: str, player: str) -> str:
    """Move paddle right in Breakout"""
    game = registry.get_game(game_id)
    if not game or game.type != "ALE/Breakout-v5":
        return f"Breakout game {game_id} not found"

    action_obj = GameAction(type="gym_step", payload={"action": 2}, player=player)
    result = game.make_move(action_obj)

    if result.success:
        base_msg = f"Moved right: {result.new_state.model_dump()}"
        return format_response_with_frame(base_msg, game_id)
    else:
        return f"Action failed: {result.error}"


@mcp.tool()
def breakout_left(game_id: str, player: str) -> str:
    """Move paddle left in Breakout"""
    game = registry.get_game(game_id)
    if not game or game.type != "ALE/Breakout-v5":
        return f"Breakout game {game_id} not found"

    action_obj = GameAction(type="gym_step", payload={"action": 3}, player=player)
    result = game.make_move(action_obj)

    if result.success:
        base_msg = f"Moved left: {result.new_state.model_dump()}"
        return format_response_with_frame(base_msg, game_id)
    else:
        return f"Action failed: {result.error}"


# Blackjack-specific tools
@mcp.tool()
def blackjack_hit(game_id: str, player: str) -> str:
    """Take another card in Blackjack"""
    game = registry.get_game(game_id)
    if not game or game.type != "Blackjack-v1":
        return f"Blackjack game {game_id} not found"

    try:
        from .games.types import GameAction
    except ImportError:
        from games.types import GameAction
    action = GameAction(
        player=player,
        type="gym_step",
        payload={"action": 1}  # 1 = Hit in Blackjack
    )

    result = game.make_move(action)

    if result.success:
        meta = result.new_state.metadata
        response = f"Hit! Your hand: {meta.get('player_hand', '?')}"
        response += f", Dealer showing: {meta.get('dealer_showing', '?')}"
        if meta.get('has_usable_ace'):
            response += " (You have a usable ace)"
        if result.new_state.status.value == "completed":
            response += f"\nGame Over: {meta.get('result', 'Game ended')}"
        return response
    else:
        return f"Hit failed: {result.error}"


@mcp.tool()
def blackjack_stand(game_id: str, player: str) -> str:
    """Keep current hand in Blackjack"""
    game = registry.get_game(game_id)
    if not game or game.type != "Blackjack-v1":
        return f"Blackjack game {game_id} not found"

    try:
        from .games.types import GameAction
    except ImportError:
        from games.types import GameAction
    action = GameAction(
        player=player,
        type="gym_step",
        payload={"action": 0}  # 0 = Stand in Blackjack
    )

    result = game.make_move(action)

    if result.success:
        meta = result.new_state.metadata
        response = f"Stand! Final hand: {meta.get('player_hand', '?')}"
        response += f"\nDealer had: {meta.get('dealer_showing', '?')}"
        response += f"\nResult: {meta.get('result', 'Game ended')}"
        response += f"\nTotal reward: {result.new_state.state.get('episode_reward', 0)}"
        return response
    else:
        return f"Stand failed: {result.error}"


@mcp.tool()
def blackjack_status(game_id: str) -> str:
    """Get current Blackjack game status"""
    game = registry.get_game(game_id)
    if not game or game.type != "Blackjack-v1":
        return f"Blackjack game {game_id} not found"

    state = game.get_state()
    meta = state.metadata

    status = f"🃏 Blackjack Status:\n"
    status += f"Your hand: {meta.get('player_hand', '?')}\n"
    status += f"Dealer showing: {meta.get('dealer_showing', '?')}\n"
    status += f"Usable ace: {'Yes' if meta.get('has_usable_ace') else 'No'}\n"
    status += f"Game status: {state.status.value}\n"

    if state.status.value == "completed":
        status += f"Result: {meta.get('result', 'Game ended')}"

    return status


async def main():
    """Main entry point with HTTP and STDIO transport support"""
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Gym MCP Server")
    parser.add_argument("--port", type=int, help="Port for HTTP transport")
    parser.add_argument("--host", type=str, default="localhost", help="Host for HTTP transport")
    parser.add_argument("--stdio", action="store_true", help="Use STDIO transport")
    parser.add_argument("--test", action="store_true", help="Test mode - quick verification")
    args = parser.parse_args()

    # Test mode for quick verification
    if args.test:
        print("🎮 Gym MCP Server - Test Mode")
        print(f"Available games: {len(registry.get_available_games())}")
        print("✅ Server configuration valid")
        return

    # Determine transport method
    port = args.port or os.getenv("PORT") or "8080"  # Default to 8080 for deployment

    if port and not args.stdio:
        # HTTP transport (for Dedalus deployment)
        print(f"🌐 Starting Gym MCP Server on HTTP {args.host}:{port}")
        print(f"🎮 Available games: {len(registry.get_available_games())}")

        # Use streamable HTTP transport
        import uvicorn
        app = mcp.streamable_http_app()

        def run_http_server():
            uvicorn.run(app, host=args.host, port=int(port))

        await asyncio.to_thread(run_http_server)
    else:
        # STDIO transport (for local Claude Desktop)
        print("🔌 Starting Gym MCP Server on STDIO transport", file=sys.stderr)
        print(f"🎮 Available games: {len(registry.get_available_games())}", file=sys.stderr)

        def run_stdio_server():
            mcp.run(transport="stdio")

        await asyncio.to_thread(run_stdio_server)


if __name__ == "__main__":
    asyncio.run(main())