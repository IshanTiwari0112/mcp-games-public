#!/usr/bin/env python3
"""
Test MCP tools from the Dedalus marketplace
"""

import asyncio

async def test_marketplace_tools():
    """Test MCP servers that are available in the Dedalus marketplace"""
    print("🛒 Testing MCP tools from Dedalus marketplace...")

    try:
        from dedalus_labs import AsyncDedalus, DedalusRunner

        # Initialize client
        client = AsyncDedalus(api_key='dsk_test_8cd724bb13c7_8d17b645d1957e1548f3ca2ad6a24f0f')
        runner = DedalusRunner(client, verbose=True)

        # Common marketplace servers to test
        marketplace_servers = [
            'brave-search',
            'filesystem',
            'sqlite',
            'postgres',
            'github',
            'slack',
            'everart',
            'memory',
            'anthropic-mcp-server',
            'web-search'
        ]

        for server in marketplace_servers:
            print(f"\n🔧 Testing {server} server...")
            try:
                response = await runner.run(
                    input=f'What tools are available from the {server} server? List the tools you can use.',
                    model='openai/gpt-4o-mini',
                    mcp_servers=[server]
                )

                print(f'✅ {server} response:')
                print(response.final_output[:300] + "..." if len(response.final_output) > 300 else response.final_output)

            except Exception as server_error:
                error_msg = str(server_error)
                if "all_mcp_servers_unavailable" in error_msg:
                    print(f'❌ {server}: Server unavailable')
                elif "not found" in error_msg.lower():
                    print(f'❌ {server}: Server not found')
                else:
                    print(f'❌ {server}: {error_msg[:100]}...')

        print("\n" + "="*60)
        print("🎯 Testing with multiple servers at once...")

        # Try multiple servers that might be available
        try:
            response = await runner.run(
                input='Show me all available tools from any MCP servers you have access to.',
                model='openai/gpt-4o-mini',
                mcp_servers=['brave-search', 'filesystem', 'memory']  # Common ones
            )

            print('🌟 Multi-server response:')
            print(response.final_output)

        except Exception as multi_error:
            print(f'❌ Multi-server test failed: {multi_error}')

        print("\n" + "="*60)
        print("📋 Testing without specifying any servers...")

        # Test without specifying servers - should show what's available by default
        response = await runner.run(
            input='What MCP servers and tools are currently available to you? Be very specific about server names and tool names.',
            model='openai/gpt-4o-mini'
        )

        print('🔍 Default available tools:')
        print(response.final_output)

    except ImportError:
        print("❌ dedalus_labs package not available")
    except Exception as e:
        print(f"❌ General error: {e}")

if __name__ == "__main__":
    asyncio.run(test_marketplace_tools())