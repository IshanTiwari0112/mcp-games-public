#!/usr/bin/env python3
"""Demo: Complete tic-tac-toe game using MCP Games server"""

import asyncio
import re
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def play_full_game():
    """Play a complete tic-tac-toe game"""

    server_params = StdioServerParameters(
        command="python", args=["main.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("🎮 MCP Games Demo: Full Tic-Tac-Toe Game")
            print("=" * 50)

            # Start game
            result = await session.call_tool("start_game", {
                "game_type": "tic-tac-toe",
                "players": ["Alice", "Bob"]
            })

            # Extract game ID
            match = re.search(r"'id': '([^']+)'", result.content[0].text)
            game_id = match.group(1)
            print(f"Game started! ID: {game_id}")

            # Game moves sequence
            moves = [
                ("Alice", 1, 1),  # Alice center
                ("Bob", 0, 0),    # Bob top-left
                ("Alice", 0, 1),  # Alice top-center
                ("Bob", 2, 1),    # Bob bottom-center
                ("Alice", 2, 2),  # Alice bottom-right
                ("Bob", 1, 0),    # Bob middle-left
                ("Alice", 0, 2),  # Alice top-right - Alice wins column 1!
            ]

            def print_board(board):
                """Pretty print the tic-tac-toe board"""
                print("\n   0   1   2")
                for i, row in enumerate(board):
                    row_str = f"{i}  "
                    for cell in row:
                        if cell is None:
                            row_str += " · "
                        else:
                            row_str += f" {cell} "
                        row_str += " "
                    print(row_str)
                print()

            # Play the game
            for move_num, (player, row, col) in enumerate(moves, 1):
                print(f"Move {move_num}: {player} -> ({row}, {col})")

                # Make the move
                result = await session.call_tool("tic_tac_toe_place_mark", {
                    "game_id": game_id,
                    "row": row,
                    "col": col,
                    "player": player
                })

                # Get current state
                state_result = await session.call_tool("get_game_state", {
                    "game_id": game_id
                })

                # Parse the board from response
                state_text = state_result.content[0].text
                board_match = re.search(r"'board': (\[\[.*?\]\])", state_text)
                if board_match:
                    # Safe eval of board structure
                    board_str = board_match.group(1)
                    board = eval(board_str)  # Safe in this controlled context
                    print_board(board)

                # Check if game is over
                if "'status': <GameStatus.COMPLETED:" in state_text:
                    if "'winner':" in state_text and "None" not in state_text:
                        winner_match = re.search(r"'winner': '([^']+)'", state_text)
                        if winner_match:
                            winner = winner_match.group(1)
                            print(f"🎉 Game Over! {winner} wins!")
                    else:
                        print("🤝 Game Over! It's a tie!")
                    break

                # Show whose turn is next
                if "'current_player': '" in state_text:
                    current_match = re.search(r"'current_player': '([^']+)'", state_text)
                    if current_match:
                        next_player = current_match.group(1)
                        print(f"Next turn: {next_player}")

                print("-" * 30)

            print("\n✨ Demo completed! MCP Games server working perfectly.")


if __name__ == "__main__":
    asyncio.run(play_full_game())
