#!/usr/bin/env python3
"""Show exactly what an AI agent sees when connecting to our MCP server"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def show_ai_perspective():
    """Show what an AI agent sees when using our MCP server"""
    print("🤖 AI Agent Perspective: Available MCP Tools")
    print("=" * 50)

    server_params = StdioServerParameters(
        command="python", args=["main.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Get all available tools (what AI sees)
            tools_result = await session.list_tools()

            print(f"🛠️  AI Agent has access to {len(tools_result.tools)} tools:")
            print()

            for i, tool in enumerate(tools_result.tools, 1):
                print(f"{i:2d}. 🔧 {tool.name}")
                print(f"    📝 {tool.description}")
                print(f"    📋 Input schema: {json.dumps(tool.inputSchema.dict() if hasattr(tool, 'inputSchema') else {}, indent=6)}")
                print()

            print("🚀 Example: AI agent usage workflow:")
            print("   1. AI calls: list_games() → sees available games")
            print("   2. AI calls: start_game(game_type='CartPole-v1', players=['AI']) → gets game_id")
            print("   3. AI calls: cartpole_move_left(game_id='abc123', player='AI') → takes action")
            print("   4. AI calls: get_game_state(game_id='abc123') → sees results")
            print("   5. Repeat steps 3-4 to play the game!")


if __name__ == "__main__":
    asyncio.run(show_ai_perspective())