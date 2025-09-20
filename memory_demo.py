#!/usr/bin/env python3
"""Demo: How AI memory and game state persistence works"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def memory_demo():
    """Show how AI maintains game state and memory"""
    print("🧠 AI Memory & Game State Demo")
    print("=" * 40)

    server_params = StdioServerParameters(
        command="python", args=["main.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("🎮 Starting a CartPole game...")
            start_result = await session.call_tool("start_game", {
                "game_type": "CartPole-v1",
                "players": ["AI_Agent"]
            })

            # Extract game ID
            import re
            game_id_match = re.search(r"'id': '([^']+)'", start_result.content[0].text)
            game_id = game_id_match.group(1)

            print(f"   Game ID: {game_id}")

            # Simulate multiple turns with memory
            for turn in range(1, 4):
                print(f"\n🔄 Turn {turn}:")

                # AI takes action
                action = "left" if turn % 2 == 1 else "right"
                tool_name = f"cartpole_move_{action}"

                result = await session.call_tool(tool_name, {
                    "game_id": game_id,
                    "player": "AI_Agent"
                })
                print(f"   ✓ AI moved {action}")

                # AI checks state (this is what gives AI memory!)
                state_result = await session.call_tool("get_game_state", {
                    "game_id": game_id
                })

                # Parse the state to show what AI "remembers"
                state_text = state_result.content[0].text

                # Extract key info that AI can see
                if "'episode_steps':" in state_text:
                    steps_match = re.search(r"'episode_steps': (\d+)", state_text)
                    steps = steps_match.group(1) if steps_match else "?"
                    print(f"   📊 AI sees: Episode step {steps}")

                if "'episode_reward':" in state_text:
                    reward_match = re.search(r"'episode_reward': ([^,}]+)", state_text)
                    reward = reward_match.group(1) if reward_match else "?"
                    print(f"   🏆 AI sees: Total reward {reward}")

                if "'observation':" in state_text:
                    print(f"   👁️  AI sees: Current cart/pole state")

            print("\n💡 How AI Memory Works:")
            print("   🔹 Server State: Game persists between tool calls")
            print("   🔹 AI Memory: Each get_game_state() refreshes AI's knowledge")
            print("   🔹 Game ID: Links all actions to the same game instance")
            print("   🔹 Context Window: AI remembers the conversation history")


if __name__ == "__main__":
    asyncio.run(memory_demo())