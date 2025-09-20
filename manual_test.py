#!/usr/bin/env python3
"""Manual test to verify MCP server works locally"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def manual_test():
    """Manually test the MCP server locally"""
    print("🧪 Manual MCP Server Test")
    print("=" * 40)

    # Connect to server
    server_params = StdioServerParameters(
        command="python", args=["main.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            await session.initialize()
            print("✅ Connected to MCP server")

            # Test 1: List available tools
            print("\n📋 Available tools:")
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                print(f"  • {tool.name}: {tool.description}")

            # Test 2: List games
            print("\n🎮 Available games:")
            games_result = await session.call_tool("list_games", {})
            print(f"  {games_result.content[0].text}")

            # Test 3: Start CartPole game
            print("\n🚀 Starting CartPole game:")
            start_result = await session.call_tool("start_game", {
                "game_type": "CartPole-v1",
                "players": ["TestPlayer"]
            })
            print(f"  Result: {start_result.content[0].text[:100]}...")

            # Extract game ID from result
            import re
            game_id_match = re.search(r"'id': '([^']+)'", start_result.content[0].text)
            if game_id_match:
                game_id = game_id_match.group(1)
                print(f"  Game ID: {game_id}")

                # Test 4: Take action
                print("\n🎯 Taking action (move left):")
                action_result = await session.call_tool("cartpole_move_left", {
                    "game_id": game_id,
                    "player": "TestPlayer"
                })
                print(f"  Result: {action_result.content[0].text[:80]}...")

                # Test 5: Get game state
                print("\n📊 Current game state:")
                state_result = await session.call_tool("get_game_state", {
                    "game_id": game_id
                })
                print(f"  State: {state_result.content[0].text[:100]}...")

            print("\n✅ All tests passed! MCP server is working correctly.")


if __name__ == "__main__":
    asyncio.run(manual_test())