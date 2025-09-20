#!/usr/bin/env python3
"""
Test using proper Dedalus pattern from the example
"""

import asyncio
import os
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv

load_dotenv()

async def test_dedalus_proper():
    """Test using the proper Dedalus pattern"""

    # Set API key as environment variable (proper way)
    os.environ['DEDALUS_API_KEY'] = 'dsk_test_8cd724bb13c7_8d17b645d1957e1548f3ca2ad6a24f0f'

    client = AsyncDedalus()  # Should pick up API key from env
    runner = DedalusRunner(client)

    print("🔍 Testing general capabilities...")

    # Test 1: General query
    response = await runner.run(
        input="What tools and capabilities do you have available? List everything you can do.",
        model="openai/gpt-4o-mini"  # Fixed model name
    )

    print("✅ General capabilities:")
    print(response.final_output)

    print("\n" + "="*60)
    print("🎮 Testing our MCP server...")

    # Test 2: Try to use our MCP server
    try:
        response = await runner.run(
            input="I want to play a game. What games are available? Use the mcp-games-public server.",
            model="openai/gpt-4o-mini",
            mcp_servers=["mcp-games-public"]
        )

        print("✅ MCP Games server response:")
        print(response.final_output)

    except Exception as e:
        print(f"❌ MCP Games server failed: {e}")

    print("\n" + "="*60)
    print("🔧 Testing tool discovery...")

    # Test 3: Ask about specific tools
    response = await runner.run(
        input="Do you have access to any game-related tools? Can you start games, make moves, or control game environments?",
        model="openai/gpt-4o-mini"
    )

    print("✅ Tool discovery:")
    print(response.final_output)

if __name__ == "__main__":
    asyncio.run(test_dedalus_proper())