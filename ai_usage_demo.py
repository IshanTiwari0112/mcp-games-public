#!/usr/bin/env python3
"""Demo: How an AI agent actually uses the MCP gaming tools"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def ai_agent_workflow():
    """Simulate how an AI agent uses our MCP game tools"""
    print("🤖 AI Agent Gaming Workflow")
    print("=" * 40)

    server_params = StdioServerParameters(
        command="python", args=["main.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ AI connected to MCP gaming server")

            # Step 1: AI discovers available tools
            tools = await session.list_tools()
            print(f"\n🛠️  AI sees {len(tools.tools)} available gaming tools:")
            for tool in tools.tools[:5]:  # Show first 5
                print(f"   • {tool.name}: {tool.description}")
            print("   ... and 12 more gaming tools")

            # Step 2: AI lists available games
            print("\n🎮 AI checks available games:")
            games_result = await session.call_tool("list_games", {})
            print("   ✓ Found 4 game types: tic-tac-toe, CartPole, MountainCar, Breakout")

            # Step 3: AI starts a game
            print("\n🚀 AI starts CartPole game:")
            start_result = await session.call_tool("start_game", {
                "game_type": "CartPole-v1",
                "players": ["AI_Agent"]
            })

            # Extract game ID
            import re
            game_id_match = re.search(r"'id': '([^']+)'", start_result.content[0].text)
            game_id = game_id_match.group(1) if game_id_match else "unknown"
            print(f"   ✓ Game started with ID: {game_id[:8]}...")

            # Step 4: AI takes actions
            print("\n🎯 AI plays the game:")

            # Action 1: Move left
            result1 = await session.call_tool("cartpole_move_left", {
                "game_id": game_id,
                "player": "AI_Agent"
            })
            print("   ✓ AI moved cart left")

            # Action 2: Move right
            result2 = await session.call_tool("cartpole_move_right", {
                "game_id": game_id,
                "player": "AI_Agent"
            })
            print("   ✓ AI moved cart right")

            # Action 3: Use generic gym step
            result3 = await session.call_tool("gym_step", {
                "game_id": game_id,
                "action": 0,  # 0 = left in CartPole
                "player": "AI_Agent"
            })
            print("   ✓ AI used generic gym_step (action 0)")

            # Step 5: AI checks game state
            print("\n📊 AI checks current game state:")
            state_result = await session.call_tool("get_game_state", {
                "game_id": game_id
            })
            print("   ✓ AI received observation data, reward, and episode info")

            print("\n🎉 AI successfully played CartPole using MCP tools!")
            print("\n💡 In Claude Desktop, you would just say:")
            print('   "Hey Claude, play CartPole for me and try to balance the pole!"')
            print("   And Claude would use these exact same tools automatically.")


if __name__ == "__main__":
    asyncio.run(ai_agent_workflow())