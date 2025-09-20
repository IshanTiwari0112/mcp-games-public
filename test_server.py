#!/usr/bin/env python3
"""Simple test client for the MCP games server"""

import asyncio
import json
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_server():
    """Test the MCP games server"""
    # Start the server
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_games.server"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            await session.initialize()

            # List available tools
            print("📋 Available tools:")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            print("\n🎮 Testing game operations...")

            # List games
            result = await session.call_tool("list_games", {})
            print(f"Available games: {result.content[0].text}")

            # Start a tic-tac-toe game
            start_result = await session.call_tool("start_game", {
                "game_type": "tic-tac-toe",
                "players": ["Alice", "Bob"]
            })
            print(f"Game started: {start_result.content[0].text}")

            # Extract game ID from the result
            import re
            game_id_match = re.search(r"'id': '([^']+)'", start_result.content[0].text)
            if game_id_match:
                game_id = game_id_match.group(1)
                print(f"Game ID: {game_id}")

                # Make a move
                move_result = await session.call_tool("make_move", {
                    "game_id": game_id,
                    "action_type": "place_mark",
                    "payload": {"row": 1, "col": 1},
                    "player": "Alice"
                })
                print(f"Move result: {move_result.content[0].text}")

                # Get game state
                state_result = await session.call_tool("get_game_state", {
                    "game_id": game_id
                })
                print(f"Game state: {state_result.content[0].text}")


if __name__ == "__main__":
    asyncio.run(test_mcp_server())