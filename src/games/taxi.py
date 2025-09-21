"""Taxi game - Pick up and drop off passengers efficiently"""

from typing import Any, Dict, List
from .gymnasium_game import GymnasiumGame
from .types import GameToolDefinition, GameResult, GameAction


class Taxi(GymnasiumGame):
    """Taxi-v3 environment - Pick up and drop off passengers in a city grid"""

    def __init__(self, game_id: str):
        super().__init__(game_id, "Taxi-v3")
        
    def _decode_state(self, obs):
        """Decode the Taxi observation into meaningful components"""
        if obs is None:
            return None
            
        # Taxi observation is encoded as a single integer
        # We need to decode it to get taxi position, passenger location, destination
        taxi_row, taxi_col, passenger_idx, destination_idx = self._decode_obs(obs)
        
        return {
            "taxi_position": (taxi_row, taxi_col),
            "passenger_location": passenger_idx,
            "destination": destination_idx
        }
    
    def _decode_obs(self, obs):
        """Decode Taxi observation (from Taxi-v3 source code logic)"""
        out = []
        out.append(obs % 5)  # taxi row
        obs = obs // 5
        out.append(obs % 5)  # taxi col  
        obs = obs // 5
        out.append(obs % 5)  # passenger location (0-3: locations, 4: in taxi)
        obs = obs // 5
        out.append(obs)      # destination
        return tuple(out)
    
    def _get_location_name(self, idx):
        """Get human-readable location names"""
        locations = {
            0: "Red (R)",
            1: "Green (G)", 
            2: "Yellow (Y)",
            3: "Blue (B)",
            4: "In Taxi"
        }
        return locations.get(idx, f"Location {idx}")
    
    def _get_text_description(self, obs):
        """Get human-readable description of current state"""
        if obs is None:
            return "Game not started"
            
        taxi_row, taxi_col, passenger_idx, destination_idx = self._decode_obs(obs)
        
        description = f"🚕 Taxi Status:\n"
        description += f"📍 Taxi at: Row {taxi_row}, Column {taxi_col}\n"
        
        if passenger_idx == 4:
            description += f"👤 Passenger: In taxi (picked up!)\n"
            description += f"🎯 Destination: {self._get_location_name(destination_idx)}\n"
            description += f"💡 Action needed: Drive to destination and drop off passenger"
        else:
            description += f"👤 Passenger at: {self._get_location_name(passenger_idx)}\n"
            description += f"🎯 Final destination: {self._get_location_name(destination_idx)}\n"
            description += f"💡 Action needed: Drive to passenger location and pick them up"
        
        return description

    def start(self, players: List[str]) -> GameResult:
        """Start Taxi game with enhanced text description"""
        result = super().start(players)
        if result.success and self.observation is not None:
            # Add human-readable state description
            decoded = self._decode_state(self.observation)
            result.new_state.metadata["text_description"] = self._get_text_description(self.observation)
            result.new_state.metadata["decoded_state"] = decoded
            result.new_state.metadata["available_actions"] = [
                "Move South", "Move North", "Move East", "Move West", 
                "Pick up passenger", "Drop off passenger"
            ]
        return result

    def make_move(self, action: GameAction) -> GameResult:
        """Make a move with enhanced feedback"""
        result = super().make_move(action)
        if result.success and self.observation is not None:
            # Add descriptive feedback
            decoded = self._decode_state(self.observation)
            result.new_state.metadata["text_description"] = self._get_text_description(self.observation)
            result.new_state.metadata["decoded_state"] = decoded
            
            # Provide action feedback
            last_reward = result.new_state.state.get("last_reward", 0)
            if last_reward == 20:
                result.new_state.metadata["action_result"] = "🎉 Successfully delivered passenger! +20 points"
                result.new_state.winner = action.player
            elif last_reward == -1:
                result.new_state.metadata["action_result"] = "❌ Invalid action (illegal pickup/dropoff)"
            elif last_reward == -10:
                result.new_state.metadata["action_result"] = "⚠️ Time penalty - each step costs -1 point"
            else:
                result.new_state.metadata["action_result"] = "✅ Action completed"
            
            # Check if game ended
            if self.terminated:
                total_reward = result.new_state.state.get("episode_reward", 0)
                if total_reward > 0:
                    result.new_state.metadata["result"] = f"🎉 Mission completed! Total score: {total_reward}"
                else:
                    result.new_state.metadata["result"] = f"Mission ended. Total score: {total_reward}"
                    
        return result

    @classmethod
    def get_tool_definitions(cls) -> List[GameToolDefinition]:
        """Taxi-specific tools"""
        base_tools = super().get_tool_definitions()
        taxi_tools = [
            GameToolDefinition(
                name="taxi_move_south",
                description="Move taxi south (down) on the grid",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"},
                        "player": {"type": "string", "description": "Player making the move"}
                    },
                    "required": ["game_id", "player"]
                }
            ),
            GameToolDefinition(
                name="taxi_move_north",
                description="Move taxi north (up) on the grid", 
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"},
                        "player": {"type": "string", "description": "Player making the move"}
                    },
                    "required": ["game_id", "player"]
                }
            ),
            GameToolDefinition(
                name="taxi_move_east",
                description="Move taxi east (right) on the grid",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"},
                        "player": {"type": "string", "description": "Player making the move"}
                    },
                    "required": ["game_id", "player"]
                }
            ),
            GameToolDefinition(
                name="taxi_move_west",
                description="Move taxi west (left) on the grid",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"},
                        "player": {"type": "string", "description": "Player making the move"}
                    },
                    "required": ["game_id", "player"]
                }
            ),
            GameToolDefinition(
                name="taxi_pickup_passenger",
                description="Pick up the passenger (only works if taxi is at passenger location)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"},
                        "player": {"type": "string", "description": "Player making the move"}
                    },
                    "required": ["game_id", "player"]
                }
            ),
            GameToolDefinition(
                name="taxi_dropoff_passenger",
                description="Drop off the passenger (only works if passenger is in taxi and at correct destination)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"},
                        "player": {"type": "string", "description": "Player making the move"}
                    },
                    "required": ["game_id", "player"]
                }
            ),
            GameToolDefinition(
                name="taxi_status",
                description="Get current Taxi game status with detailed city map and objectives",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"}
                    },
                    "required": ["game_id"]
                }
            )
        ]
        return base_tools + taxi_tools

    @classmethod
    def get_game_info(cls) -> Dict[str, Any]:
        return {
            "type": "Taxi-v3",
            "name": "Taxi Driver",
            "description": "Navigate a city grid to pick up and drop off passengers efficiently. Great for route planning and logistics!",
            "min_players": 1,
            "max_players": 1,
            "custom_tools": [
                "taxi_move_south", "taxi_move_north", "taxi_move_east", "taxi_move_west",
                "taxi_pickup_passenger", "taxi_dropoff_passenger", "taxi_status",
                "gym_step", "gym_reset"
            ],
            "action_space": "Discrete(6) - 0: South, 1: North, 2: East, 3: West, 4: Pickup, 5: Dropoff",
            "observation_space": "Discrete(500) - Encoded state of taxi position, passenger location, destination",
            "rules": [
                "Navigate 5x5 city grid to pick up and drop off passengers",
                "Pick up passenger at their location (Red, Green, Yellow, Blue)",
                "Drop off passenger at their destination for +20 reward", 
                "Each step costs -1 point (time penalty)",
                "Illegal pickup/dropoff actions cost -10 points"
            ],
            "strategy_tips": [
                "Plan efficient routes to minimize steps",
                "Pick up passenger first, then navigate to destination",
                "Watch for walls - not all grid positions are accessible",
                "Goal is to maximize reward by completing trips quickly"
            ]
        }
