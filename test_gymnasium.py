#!/usr/bin/env python3
"""Test Gymnasium integration with the MCP server"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_gymnasium():
    """Test Gymnasium games integration"""

    server_params = StdioServerParameters(
        command="python", args=["main.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ Connected to Gym MCP server")

            # List all available tools
            tools_result = await session.list_tools()
            print(f"\n🛠️ Available tools ({len(tools_result.tools)}):")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")

            # List available games
            print("\n📋 Available games:")
            result = await session.call_tool("list_games", {})
            print(f"  {result.content[0].text}")

            # Test CartPole
            print("\n🎮 Testing CartPole-v1:")

            # Start CartPole game
            start_result = await session.call_tool("start_game", {
                "game_type": "CartPole-v1",
                "players": ["AI_Agent"]
            })
            print(f"  Start result: {start_result.content[0].text[:100]}...")

            # Extract game ID
            import re
            game_id_match = re.search(r"'id': '([^']+)'", start_result.content[0].text)
            if game_id_match:
                game_id = game_id_match.group(1)
                print(f"  Game ID: {game_id}")

                # Take a few actions
                print("\n🏃 Taking actions:")

                # Move left
                result = await session.call_tool("cartpole_move_left", {
                    "game_id": game_id,
                    "player": "AI_Agent"
                })
                print(f"  Left: {result.content[0].text[:80]}...")

                # Move right
                result = await session.call_tool("cartpole_move_right", {
                    "game_id": game_id,
                    "player": "AI_Agent"
                })
                print(f"  Right: {result.content[0].text[:80]}...")

                # Use generic gym_step
                result = await session.call_tool("gym_step", {
                    "game_id": game_id,
                    "action": 0,
                    "player": "AI_Agent"
                })
                print(f"  Generic step: {result.content[0].text[:80]}...")

                # Get final state
                state_result = await session.call_tool("get_game_state", {
                    "game_id": game_id
                })
                print(f"  Final state: {state_result.content[0].text[:100]}...")

            # Test MountainCar
            print("\n🏔️ Testing MountainCar-v0:")

            # Start MountainCar
            start_result = await session.call_tool("start_game", {
                "game_type": "MountainCar-v0",
                "players": ["AI_Agent"]
            })
            print(f"  Start result: {start_result.content[0].text[:100]}...")

            # Extract game ID
            game_id_match = re.search(r"'id': '([^']+)'", start_result.content[0].text)
            if game_id_match:
                game_id = game_id_match.group(1)

                # Push left
                result = await session.call_tool("mountain_car_push_left", {
                    "game_id": game_id,
                    "player": "AI_Agent"
                })
                print(f"  Push left: {result.content[0].text[:80]}...")

            print("\n✅ Gymnasium integration test completed!")


if __name__ == "__main__":
    asyncio.run(test_gymnasium())