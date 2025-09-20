#!/usr/bin/env python3
"""
Run MCP Games Server with Web Viewer
This starts both the MCP server and web viewer for visual game display
"""

import asyncio
import threading
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_web_viewer():
    """Run the web viewer in a separate thread"""
    from src.web_viewer import run_web_viewer
    run_web_viewer(port=8000)


async def run_mcp_server():
    """Run the MCP server"""
    from src.main import main
    await main()


async def main():
    """Run both servers together"""
    print("🚀 Starting MCP Games with Visual Web Viewer")
    print("=" * 50)

    # Start web viewer in background thread
    viewer_thread = threading.Thread(target=run_web_viewer, daemon=True)
    viewer_thread.start()

    # Give viewer time to start
    time.sleep(2)

    print("\n📋 Instructions:")
    print("1. MCP Server running (for Claude/AI agents)")
    print("2. Web Viewer at: http://localhost:8000")
    print("3. Start a game via Claude, copy the game ID")
    print("4. Paste game ID in web viewer to watch live!\n")

    # Run MCP server (this blocks)
    await run_mcp_server()


if __name__ == "__main__":
    asyncio.run(main())