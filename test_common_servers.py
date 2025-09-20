#!/usr/bin/env python3
"""
Test common MCP servers that are typically available
"""

import asyncio

async def test_common_servers():
    """Test common MCP servers"""
    print("🔧 Testing common MCP servers...")

    try:
        from dedalus_labs import AsyncDedalus, DedalusRunner

        client = AsyncDedalus(api_key='dsk_test_8cd724bb13c7_8d17b645d1957e1548f3ca2ad6a24f0f')
        runner = DedalusRunner(client, verbose=True)

        # Common server names based on typical MCP ecosystems
        servers_to_test = [
            'filesystem-server',
            'web-search-server',
            'brave-search-server',
            'github-server',
            'sqlite-server',
            'postgres-server',
            'slack-server',
            'memory-server',
            'anthropic-server',
            'fetch-server',
            'time-server',
            'weather-server',
            'calculator-server',
            'shell-server',
            'docker-server'
        ]

        working_servers = []

        for server in servers_to_test:
            print(f"\n🔍 Testing {server}...")
            try:
                response = await runner.run(
                    input=f'Use any tools available from {server} to show me what you can do.',
                    model='openai/gpt-4o-mini',
                    mcp_servers=[server]
                )

                print(f'✅ {server} WORKS!')
                print(f'Response: {response.final_output[:200]}...')
                working_servers.append(server)

            except Exception as e:
                error_str = str(e)
                if "all_mcp_servers_unavailable" in error_str:
                    print(f'❌ {server}: Unavailable')
                elif "not found" in error_str.lower() or "404" in error_str:
                    print(f'❌ {server}: Not found')
                else:
                    print(f'❌ {server}: {error_str[:50]}...')

        print(f"\n🎉 Working servers found: {working_servers}")

        if working_servers:
            print(f"\n🚀 Testing with working server: {working_servers[0]}")
            response = await runner.run(
                input='Show me all the tools and capabilities you have. List every function available.',
                model='openai/gpt-4o-mini',
                mcp_servers=[working_servers[0]]
            )
            print(f'Full capabilities: {response.final_output}')

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_common_servers())