#!/usr/bin/env python3
"""Quick test of a few specific servers"""

import asyncio

async def quick_test():
    """Quick test of specific servers"""
    try:
        from dedalus_labs import AsyncDedalus, DedalusRunner

        client = AsyncDedalus(api_key='dsk_test_8cd724bb13c7_8d17b645d1957e1548f3ca2ad6a24f0f')
        runner = DedalusRunner(client, verbose=False)

        # Test just a few likely candidates
        test_servers = ['filesystem', 'web-search', 'brave-search']

        for server in test_servers:
            print(f"Testing {server}...")
            try:
                response = await runner.run(
                    input='What tools are available?',
                    model='openai/gpt-4o-mini',
                    mcp_servers=[server]
                )
                print(f"✅ {server} works: {response.final_output[:100]}...")
                break  # Stop at first working one
            except Exception as e:
                print(f"❌ {server}: failed")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(quick_test())