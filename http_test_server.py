#!/usr/bin/env python3
"""HTTP wrapper for testing MCP server with curl"""

import asyncio
import json
import subprocess
from typing import Dict, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="MCP Games HTTP Test Server")

async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Call an MCP tool and return the result"""
    server_params = StdioServerParameters(
        command="python", args=["main.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return result.content[0].text

@app.get("/")
async def root():
    return {"message": "MCP Games HTTP Test Server", "endpoints": [
        "GET /tools - List available tools",
        "GET /games - List available games", 
        "POST /games/start - Start a new game",
        "POST /games/{game_id}/action - Take action in game",
        "GET /games/{game_id}/state - Get game state"
    ]}

@app.get("/tools")
async def list_tools():
    """List available MCP tools"""
    try:
        result = await call_mcp_tool("list_games", {})
        return {"tools": ["list_games", "start_game", "make_move", "get_game_state", "reset_game"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/games")
async def list_games():
    """List available games"""
    try:
        result = await call_mcp_tool("list_games", {})
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/games/start")
async def start_game(game_type: str, players: list):
    """Start a new game"""
    try:
        result = await call_mcp_tool("start_game", {
            "game_type": game_type,
            "players": players
        })
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/games/{game_id}/action")
async def take_action(game_id: str, action_type: str, payload: dict, player: str):
    """Take action in a game"""
    try:
        result = await call_mcp_tool("make_move", {
            "game_id": game_id,
            "action_type": action_type,
            "payload": payload,
            "player": player
        })
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/games/{game_id}/state")
async def get_game_state(game_id: str):
    """Get current game state"""
    try:
        result = await call_mcp_tool("get_game_state", {"game_id": game_id})
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
