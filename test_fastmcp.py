#!/usr/bin/env python3
"""Test the FastMCP games server"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_server():
    """Test the FastMCP games server"""

    # Connect to our server
    server_params = StdioServerParameters(
        command="python",
        args=["main.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            print("✅ Connected to MCP Games server")

            # List tools
            tools_result = await session.list_tools()
            print(f"\n🛠️ Available tools ({len(tools_result.tools)}):")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Test list_games
            print("\n📋 Testing list_games:")
            result = await session.call_tool("list_games", {})
            print(f"  Result: {result.content[0].text}")

            # Test start_game
            print("\n🎮 Starting tic-tac-toe game:")
            result = await session.call_tool("start_game", {
                "game_type": "tic-tac-toe",
                "players": ["Alice", "Bob"]
            })
            print(f"  Result: {result.content[0].text}")

            # Extract game ID
            result_text = result.content[0].text
            if "'id':" in result_text:
                # Parse game ID from the response
                import re
                match = re.search(r"'id': '([^']+)'", result_text)
                if match:
                    game_id = match.group(1)
                    print(f"  Game ID: {game_id}")

                    # Test make_move
                    print("\n🎯 Making a move:")
                    result = await session.call_tool("make_move", {
                        "game_id": game_id,
                        "action_type": "place_mark",
                        "payload": {"row": 1, "col": 1},
                        "player": "Alice"
                    })
                    print(f"  Result: {result.content[0].text}")

                    # Test get_game_state
                    print("\n📊 Getting game state:")
                    result = await session.call_tool("get_game_state", {
                        "game_id": game_id
                    })
                    print(f"  Result: {result.content[0].text}")

                    # Test tic_tac_toe_place_mark (game-specific tool)
                    print("\n⭕ Using game-specific tool:")
                    result = await session.call_tool("tic_tac_toe_place_mark", {
                        "game_id": game_id,
                        "row": 0,
                        "col": 0,
                        "player": "Bob"
                    })
                    print(f"  Result: {result.content[0].text}")

            print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_server())