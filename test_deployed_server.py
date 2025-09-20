#!/usr/bin/env python3
"""
Test the deployed MCP server via Dedalus
"""

import asyncio

async def test_deployed_mcp_server():
    """Test the deployed MCP server using Dedalus client"""
    print("🔌 Testing deployed MCP server via Dedalus...")

    try:
        # Import Dedalus client
        from dedalus_labs import AsyncDedalus, DedalusRunner

        # Initialize client
        client = AsyncDedalus(api_key='dsk_test_8cd724bb13c7_8d17b645d1957e1548f3ca2ad6a24f0f')
        runner = DedalusRunner(client, verbose=True)

        print('🔍 Testing connection to mcp-games-public...')

        # Test 1: List all available tools
        print("\n📋 Test 1: Listing all tools...")
        response = await runner.run(
            input='List all available tools and functions. Show me exactly what tools you have access to.',
            model='openai/gpt-4o-mini',
            mcp_servers=['mcp-games-public']
        )

        print('✅ Response received:')
        print(response.final_output)

        # Test 2: Try to start a game
        print("\n🎮 Test 2: Trying to start a CartPole game...")
        response2 = await runner.run(
            input='Use the start_game tool to start a CartPole-v1 game with player name "test_player"',
            model='openai/gpt-4o-mini',
            mcp_servers=['mcp-games-public']
        )

        print('🎯 Game start response:')
        print(response2.final_output)

        # Test 3: Try to make a move if game started
        print("\n🕹️ Test 3: Trying to make a move...")
        response3 = await runner.run(
            input='If a game was started, use cartpole_move_left to move the cart left. Show me the visual frame if available.',
            model='openai/gpt-4o-mini',
            mcp_servers=['mcp-games-public']
        )

        print('🎬 Move response:')
        print(response3.final_output)

        print("\n🎉 All tests completed!")

    except ImportError:
        print("❌ dedalus_labs package not available. Install with: pip install dedalus_labs")
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_deployed_mcp_server())