#!/usr/bin/env python3
"""
Simple script to list all available MCP tools from any server via Dedalus
"""

import asyncio

async def list_all_available_tools():
    """List all available MCP tools through Dedalus"""
    print("🔍 Listing ALL available MCP tools via Dedalus...")

    try:
        from dedalus_labs import AsyncDedalus, DedalusRunner

        # Initialize client
        client = AsyncDedalus(api_key='dsk_test_8cd724bb13c7_8d17b645d1957e1548f3ca2ad6a24f0f')
        runner = DedalusRunner(client, verbose=True)

        print('📋 Requesting list of all available tools...')

        # Simple request to list all tools without specifying servers
        response = await runner.run(
            input='What tools and functions are available to you? List everything you have access to.',
            model='openai/gpt-4o-mini'
            # Note: No mcp_servers specified - should show all available
        )

        print('✅ Response:')
        print(response.final_output)

        print("\n" + "="*60)
        print("🎯 Trying with specific server request...")

        # Try specifically asking for mcp-games-public
        try:
            response2 = await runner.run(
                input='Show me tools from the mcp-games-public server specifically.',
                model='openai/gpt-4o-mini',
                mcp_servers=['mcp-games-public']
            )

            print('🎮 mcp-games-public response:')
            print(response2.final_output)

        except Exception as specific_error:
            print(f"❌ Specific server request failed: {specific_error}")

    except ImportError:
        print("❌ dedalus_labs package not available. Install with: pip install dedalus_labs")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(list_all_available_tools())