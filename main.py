#!/usr/bin/env python3
"""
MCP Games Server - Root Entry Point

This is the main entry point expected by Dedalus Labs conventions.
It imports and runs the actual server implementation from src/main.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Debug info for troubleshooting
print(f"DEBUG: Starting MCP server from {__file__}", file=sys.stderr)
print(f"DEBUG: Working directory: {os.getcwd()}", file=sys.stderr)
print(f"DEBUG: Python path: {sys.path[:3]}...", file=sys.stderr)

# Ensure src is in Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Change to the project directory to ensure relative imports work
os.chdir(Path(__file__).parent)

async def main():
    """Main entry point"""
    # Import here to avoid import issues
    from src.main import main as server_main
    await server_main()


if __name__ == "__main__":
    asyncio.run(main())