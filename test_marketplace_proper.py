#!/usr/bin/env python3
"""
Test marketplace MCP servers using proper Dedalus pattern
"""

import asyncio
import os
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv

load_dotenv()

async def test_marketplace_servers():
    """Test marketplace MCP servers with proper pattern"""

    # Set API key as environment variable
    os.environ['DEDALUS_API_KEY'] = 'dsk_test_8cd724bb13c7_8d17b645d1957e1548f3ca2ad6a24f0f'

    client = AsyncDedalus()
    runner = DedalusRunner(client)

    # Common marketplace server names to test
    marketplace_servers = [
        "brave-search",
        "filesystem",
        "sqlite",
        "postgres",
        "github",
        "slack",
        "memory",
        "weather",
        "time",
        "calculator",
        "web-search",
        "anthropic",
        "openai",
        "claude",
        "fetch",
        "http",
        "shell",
        "docker",
        "kubernetes",
        "aws",
        "gcp",
        "azure"
    ]

    working_servers = []

    for server in marketplace_servers:
        print(f"🔍 Testing {server}...")

        try:
            response = await runner.run(
                input=f"What tools are available? Show me what you can do with {server}.",
                model="openai/gpt-4o-mini",
                mcp_servers=[server]
            )

            print(f"✅ {server} WORKS!")
            print(f"Response: {response.final_output[:150]}...")
            working_servers.append(server)

            # Don't test all if we find working ones - save time
            if len(working_servers) >= 3:
                print("Found enough working servers, stopping search...")
                break

        except Exception as e:
            error_str = str(e)
            if "all_mcp_servers_unavailable" in error_str:
                print(f"❌ {server}: Server unavailable")
            elif "not found" in error_str.lower():
                print(f"❌ {server}: Server not found")
            elif "502" in error_str:
                print(f"❌ {server}: Server error")
            else:
                print(f"❌ {server}: {error_str[:60]}...")

    print(f"\n🎉 Working servers: {working_servers}")

    if working_servers:
        print(f"\n🚀 Testing capabilities of {working_servers[0]}...")

        try:
            response = await runner.run(
                input="List all available tools and functions. Show me everything you can do.",
                model="openai/gpt-4o-mini",
                mcp_servers=[working_servers[0]]
            )

            print(f"Full capabilities of {working_servers[0]}:")
            print(response.final_output)

        except Exception as e:
            print(f"Error getting capabilities: {e}")

    else:
        print("\n❌ No working marketplace servers found")

        # Test without any MCP servers to see baseline
        print("\n🔍 Testing baseline capabilities (no MCP servers)...")
        response = await runner.run(
            input="What can you do? What tools and capabilities do you have available?",
            model="openai/gpt-4o-mini"
        )

        print("Baseline capabilities:")
        print(response.final_output)

if __name__ == "__main__":
    asyncio.run(test_marketplace_servers())