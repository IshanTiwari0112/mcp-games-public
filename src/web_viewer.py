#!/usr/bin/env python3
"""
Web viewer for MCP Games - displays games visually in browser
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import Dict, Optional
import asyncio
import json
import base64
import numpy as np
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active game connections
active_games: Dict[str, dict] = {}


@app.get("/")
async def home():
    """Serve the game viewer HTML"""
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>MCP Games Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: white;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }

        #game-container {
            border: 2px solid #444;
            padding: 20px;
            border-radius: 10px;
            background: #222;
        }

        #game-canvas {
            border: 1px solid #666;
            image-rendering: pixelated;
            width: 480px;
            height: 630px;
        }

        #stats {
            margin-top: 20px;
            padding: 10px;
            background: #333;
            border-radius: 5px;
        }

        #controls {
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }

        button {
            padding: 10px 20px;
            font-size: 16px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }

        button:hover {
            background: #45a049;
        }

        #connection-status {
            padding: 5px 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }

        .connected {
            background: green;
        }

        .disconnected {
            background: red;
        }

        #game-selector {
            margin-bottom: 20px;
        }

        select {
            padding: 5px;
            font-size: 16px;
            background: #333;
            color: white;
            border: 1px solid #666;
        }
    </style>
</head>
<body>
    <h1>🎮 MCP Games Live Viewer</h1>

    <div id="connection-status" class="disconnected">
        Disconnected
    </div>

    <div id="game-selector">
        <label for="game-id">Game ID: </label>
        <input type="text" id="game-id" placeholder="Enter game ID from MCP">
        <button onclick="connectToGame()">Connect</button>
    </div>

    <div id="game-container">
        <canvas id="game-canvas" width="160" height="210"></canvas>

        <div id="stats">
            <div>Score: <span id="score">0</span></div>
            <div>Steps: <span id="steps">0</span></div>
            <div>Status: <span id="status">Not Connected</span></div>
            <div>Game Type: <span id="game-type">-</span></div>
        </div>

        <div id="controls">
            <button onclick="sendAction('left')">← Left</button>
            <button onclick="sendAction('fire')">🔥 Fire</button>
            <button onclick="sendAction('right')">Right →</button>
            <button onclick="sendAction('noop')">Wait</button>
        </div>
    </div>

    <script>
        let ws = null;
        let currentGameId = null;

        function connectToGame() {
            const gameId = document.getElementById('game-id').value;
            if (!gameId) {
                alert('Please enter a game ID');
                return;
            }

            if (ws) {
                ws.close();
            }

            currentGameId = gameId;
            ws = new WebSocket(`ws://localhost:8000/ws/${gameId}`);

            ws.onopen = function() {
                document.getElementById('connection-status').className = 'connected';
                document.getElementById('connection-status').textContent = 'Connected';
                console.log('Connected to game:', gameId);
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateGameDisplay(data);
            };

            ws.onclose = function() {
                document.getElementById('connection-status').className = 'disconnected';
                document.getElementById('connection-status').textContent = 'Disconnected';
            };

            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }

        function updateGameDisplay(data) {
            // Update stats
            document.getElementById('score').textContent = data.score || 0;
            document.getElementById('steps').textContent = data.steps || 0;
            document.getElementById('status').textContent = data.status || 'Unknown';
            document.getElementById('game-type').textContent = data.game_type || 'Unknown';

            // Update canvas if image data present
            if (data.image) {
                const canvas = document.getElementById('game-canvas');
                const ctx = canvas.getContext('2d');
                const img = new Image();

                img.onload = function() {
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                };

                img.src = 'data:image/png;base64,' + data.image;
            }
        }

        function sendAction(action) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    action: action,
                    game_id: currentGameId
                }));
            } else {
                alert('Not connected to a game');
            }
        }
    </script>
</body>
</html>
    """)


@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    """WebSocket endpoint for real-time game updates"""
    await websocket.accept()

    try:
        while True:
            # Get game state from MCP server
            game_state = get_game_state_from_mcp(game_id)

            if game_state:
                # Send update to browser
                await websocket.send_json(game_state)

            # Wait a bit before next update
            await asyncio.sleep(0.1)  # 10 FPS

    except WebSocketDisconnect:
        print(f"Client disconnected from game {game_id}")


def get_game_state_from_mcp(game_id: str) -> Optional[dict]:
    """
    Get current game state from MCP server
    This would connect to the MCP server's game registry
    """
    # Import the registry from the MCP server
    try:
        from games.registry import GameRegistry
        registry = GameRegistry()

        game = registry.get_game(game_id)
        if not game:
            return None

        state = game.get_state()

        # Extract and process the game data
        result = {
            "game_id": game_id,
            "game_type": state.type,
            "status": state.status.value,
            "score": state.state.get("episode_reward", 0),
            "steps": state.state.get("episode_steps", 0),
        }

        # Convert image data if present
        obs = state.state.get("observation")
        if obs and obs.get("type") == "image":
            # Decode base64 to image
            img_data = base64.b64decode(obs["data"])
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img_array = img_array.reshape(obs["shape"])

            # Convert to PNG for browser display
            img = Image.fromarray(img_array)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            result["image"] = img_base64

        return result
    except Exception as e:
        print(f"Error getting game state: {e}")
        return None


@app.get("/api/games")
async def list_games():
    """List all active games"""
    from games.registry import GameRegistry
    registry = GameRegistry()

    games = registry.get_active_games()
    return {"games": [
        {
            "id": game_id,
            "type": game.type,
            "status": game.state.status.value
        }
        for game_id, game in games.items()
    ]}


def run_web_viewer(port: int = 8000):
    """Run the web viewer server"""
    print(f"🌐 Starting MCP Games Web Viewer on http://localhost:{port}")
    print(f"📋 Open in browser to view games visually")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run_web_viewer()