#!/usr/bin/env python3
"""Quick test for Atari Breakout"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_breakout():
    """Test Atari Breakout"""

    server_params = StdioServerParameters(
        command="python", args=["main.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("🕹️ Testing Atari Breakout")

            # Start Breakout
            start_result = await session.call_tool("start_game", {
                "game_type": "ALE/Breakout-v5",
                "players": ["AI_Agent"]
            })
            print(f"Start result: {start_result.content[0].text[:100]}...")

            # Extract game ID
            import re
            game_id_match = re.search(r"'id': '([^']+)'", start_result.content[0].text)
            if game_id_match:
                game_id = game_id_match.group(1)

                # Fire the ball
                result = await session.call_tool("breakout_fire", {
                    "game_id": game_id,
                    "player": "AI_Agent"
                })
                print(f"Fire: {result.content[0].text[:80]}...")

                # Move paddle
                result = await session.call_tool("breakout_right", {
                    "game_id": game_id,
                    "player": "AI_Agent"
                })
                print(f"Move right: {result.content[0].text[:80]}...")

            print("✅ Breakout test completed!")


if __name__ == "__main__":
    asyncio.run(test_breakout())