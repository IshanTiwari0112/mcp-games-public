#!/usr/bin/env python3
"""Simple demo showing Alice winning tic-tac-toe"""

import asyncio
import re
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def simple_win_demo():
    """Demo Alice winning with top row"""

    server_params = StdioServerParameters(
        command="python", args=["main.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("🎮 Simple Demo: Alice wins with top row")
            print("=" * 40)

            # Start game
            result = await session.call_tool("start_game", {
                "game_type": "tic-tac-toe",
                "players": ["Alice", "Bob"]
            })

            # Extract game ID
            match = re.search(r"'id': '([^']+)'", result.content[0].text)
            game_id = match.group(1)
            print(f"Game ID: {game_id}\n")

            # Simple winning sequence - Alice gets top row
            moves = [
                ("Alice", 0, 0),  # Alice top-left
                ("Bob", 1, 0),    # Bob middle-left
                ("Alice", 0, 1),  # Alice top-center
                ("Bob", 1, 1),    # Bob center
                ("Alice", 0, 2),  # Alice top-right - Alice wins!
            ]

            def print_board_from_state(state_text):
                """Extract and print board from game state"""
                board_match = re.search(r"'board': (\[\[.*?\]\])", state_text)
                if board_match:
                    board_str = board_match.group(1)
                    board = eval(board_str)  # Safe in controlled context
                    print("   0   1   2")
                    for i, row in enumerate(board):
                        row_str = f"{i}  "
                        for cell in row:
                            row_str += f" {cell or '·'} "
                        print(row_str)
                    print()

            # Play moves
            for move_num, (player, row, col) in enumerate(moves, 1):
                print(f"Move {move_num}: {player} places at ({row}, {col})")

                # Make move
                result = await session.call_tool("tic_tac_toe_place_mark", {
                    "game_id": game_id,
                    "row": row,
                    "col": col,
                    "player": player
                })

                print(f"Move result: {result.content[0].text[:50]}...")

                # Get state
                state_result = await session.call_tool("get_game_state", {
                    "game_id": game_id
                })

                state_text = state_result.content[0].text
                print_board_from_state(state_text)

                # Check for game completion
                if "COMPLETED" in state_text:
                    winner_match = re.search(r"'winner': '([^']+)'", state_text)
                    if winner_match:
                        winner = winner_match.group(1)
                        print(f"🎉 {winner} WINS!")
                    else:
                        print("🤝 It's a tie!")
                    break

                print("-" * 30)

            print("✅ Demo completed!")


if __name__ == "__main__":
    asyncio.run(simple_win_demo())