#!/usr/bin/env python3
"""Show exactly what state information AI receives"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def show_ai_state_view():
    """Show the exact state data AI receives"""
    print("👁️  What AI Actually Sees: Complete Game State")
    print("=" * 50)

    server_params = StdioServerParameters(
        command="python", args=["main.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Start game
            start_result = await session.call_tool("start_game", {
                "game_type": "CartPole-v1",
                "players": ["AI"]
            })

            # Extract game ID
            import re
            game_id_match = re.search(r"'id': '([^']+)'", start_result.content[0].text)
            game_id = game_id_match.group(1)

            print(f"🎮 Game ID: {game_id}")
            print(f"🚀 Initial state AI sees:")
            print(f"   {start_result.content[0].text[:200]}...")

            # Take action
            print(f"\n🎯 AI takes action (move left)...")
            action_result = await session.call_tool("cartpole_move_left", {
                "game_id": game_id,
                "player": "AI"
            })

            # Get full state - this is what gives AI its "memory"
            print(f"\n📊 Complete state AI receives after action:")
            state_result = await session.call_tool("get_game_state", {
                "game_id": game_id
            })

            # Parse and display the structured state
            state_text = state_result.content[0].text
            print("=" * 50)
            print(state_text)
            print("=" * 50)

            print(f"\n🧠 AI Memory Components:")
            print(f"   🆔 Game ID: Persistent across all tool calls")
            print(f"   📈 Episode Steps: Tracks game progression")
            print(f"   🏆 Cumulative Reward: Running score")
            print(f"   📊 Observation: Current environment state")
            print(f"   ℹ️  Info: Additional environment data")
            print(f"   🎭 Status: Game active/completed")

            print(f"\n💬 How AI Uses This:")
            print(f"   1. AI calls get_game_state() to refresh memory")
            print(f"   2. AI analyzes observation data for strategy")
            print(f"   3. AI tracks rewards to measure performance")
            print(f"   4. AI continues playing based on current state")


if __name__ == "__main__":
    asyncio.run(show_ai_state_view())