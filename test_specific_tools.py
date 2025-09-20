#!/usr/bin/env python3
"""
Test specific MCP tools that should be available on Dedalus
"""

import asyncio

async def test_specific_tools():
    """Test specific tools that are known to be available on Dedalus"""
    print("🔍 Testing specific MCP tools on Dedalus...")

    try:
        from dedalus_labs import AsyncDedalus, DedalusRunner

        # Initialize client
        client = AsyncDedalus(api_key='dsk_test_8cd724bb13c7_8d17b645d1957e1548f3ca2ad6a24f0f')
        runner = DedalusRunner(client, verbose=True)

        # Test common/built-in tools
        tools_to_test = [
            "web_search",
            "browser",
            "search_web",
            "filesystem",
            "file_operations",
            "code_execution",
            "python",
            "terminal",
            "bash"
        ]

        for tool in tools_to_test:
            print(f"\n🔧 Testing {tool}...")
            try:
                response = await runner.run(
                    input=f'Use the {tool} tool to show me it\'s working. Just demonstrate that you have access to {tool}.',
                    model='openai/gpt-4o-mini'
                )

                print(f'✅ {tool} response:')
                print(response.final_output[:200] + "..." if len(response.final_output) > 200 else response.final_output)

            except Exception as tool_error:
                print(f'❌ {tool} failed: {str(tool_error)[:100]}...')

        print("\n" + "="*60)
        print("📋 Trying general capability query...")

        # General capability test
        response = await runner.run(
            input='What can you help me with? What tools, functions, or capabilities do you have access to? Be specific about any special tools or MCP servers.',
            model='openai/gpt-4o-mini'
        )

        print('🎯 General capabilities:')
        print(response.final_output)

    except ImportError:
        print("❌ dedalus_labs package not available")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_specific_tools())