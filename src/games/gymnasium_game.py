"""Base class for Gymnasium environment games"""

import json
import base64
from typing import Any, Dict, List, Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass

from .types import Game, GameAction, GameResult, GameState, GameStatus, GameToolDefinition


class GymnasiumGame(Game):
    """Base class for wrapping Gymnasium environments as MCP games"""

    def __init__(self, game_id: str, env_name: str):
        super().__init__(game_id, env_name)
        self.env_name = env_name
        self.env = None
        self.observation = None
        self.info = None
        self.terminated = False
        self.truncated = False
        self.episode_steps = 0
        self.episode_reward = 0.0

    def _create_env(self):
        """Create the Gymnasium environment"""
        if self.env is None:
            self.env = gym.make(self.env_name)

    def start(self, players: List[str]) -> GameResult:
        """Start the environment - single player for most Gym environments"""
        if len(players) != 1:
            return GameResult(
                success=False,
                error=f"{self.env_name} is a single-player environment"
            )

        try:
            self._create_env()
            self.observation, self.info = self.env.reset()

            self.state.players = players
            self.state.current_player = players[0]
            self.state.status = GameStatus.ACTIVE
            self.state.moves = 0
            self.episode_steps = 0
            self.episode_reward = 0.0
            self.terminated = False
            self.truncated = False

            # Store the initial observation
            self.state.state = {
                "observation": self._serialize_observation(self.observation),
                "info": self._serialize_info(self.info),
                "episode_reward": self.episode_reward,
                "episode_steps": self.episode_steps,
                "action_space": self._describe_action_space(),
                "observation_space": self._describe_observation_space(),
            }

            return GameResult(success=True, new_state=self.state)

        except Exception as e:
            return GameResult(success=False, error=f"Failed to start environment: {str(e)}")

    def make_move(self, action: GameAction) -> GameResult:
        """Execute an action in the environment"""
        if self.state.status != GameStatus.ACTIVE:
            return GameResult(success=False, error="Game is not active")

        if self.terminated or self.truncated:
            return GameResult(success=False, error="Episode has ended")

        if action.player != self.state.current_player:
            return GameResult(success=False, error="Wrong player")

        # Accept both "gym_step" and direct action types
        if action.type not in ["gym_step", "step"]:
            return GameResult(success=False, error=f"Invalid action type: {action.type}")

        try:
            # Parse the action
            gym_action = self._parse_action(action.payload)

            # Execute the action
            self.observation, reward, self.terminated, self.truncated, self.info = self.env.step(gym_action)

            # Update game state
            self.episode_steps += 1
            self.episode_reward += reward
            self.state.moves += 1

            # Update state
            self.state.state.update({
                "observation": self._serialize_observation(self.observation),
                "info": self._serialize_info(self.info),
                "episode_reward": self.episode_reward,
                "episode_steps": self.episode_steps,
                "last_reward": float(reward),
                "terminated": self.terminated,
                "truncated": self.truncated,
            })

            # Check if episode is done
            if self.terminated or self.truncated:
                self.state.status = GameStatus.COMPLETED
                if self.terminated:
                    self.state.metadata["end_reason"] = "terminated"
                else:
                    self.state.metadata["end_reason"] = "truncated"

            return GameResult(success=True, new_state=self.state)

        except Exception as e:
            return GameResult(success=False, error=f"Action failed: {str(e)}")

    def get_state(self) -> GameState:
        """Get current game state"""
        return self.state

    def reset(self) -> GameResult:
        """Reset the environment"""
        try:
            if self.env is not None:
                self.observation, self.info = self.env.reset()

            self.state.status = GameStatus.WAITING
            self.state.moves = 0
            self.episode_steps = 0
            self.episode_reward = 0.0
            self.terminated = False
            self.truncated = False

            self.state.state = {
                "observation": self._serialize_observation(self.observation) if self.observation is not None else None,
                "info": self._serialize_info(self.info) if self.info is not None else None,
                "episode_reward": 0.0,
                "episode_steps": 0,
            }

            return GameResult(success=True, new_state=self.state)

        except Exception as e:
            return GameResult(success=False, error=f"Reset failed: {str(e)}")

    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Get available actions based on action space"""
        if self.env is None:
            return []

        actions = []
        action_space = self.env.action_space

        if isinstance(action_space, Discrete):
            for i in range(action_space.n):
                actions.append({
                    "type": "discrete",
                    "value": i,
                    "description": f"Discrete action {i}"
                })
        elif isinstance(action_space, Box):
            actions.append({
                "type": "continuous",
                "shape": action_space.shape,
                "low": action_space.low.tolist(),
                "high": action_space.high.tolist(),
                "description": f"Continuous action in space {action_space.shape}"
            })

        return actions

    def _serialize_observation(self, obs) -> Union[List, Dict, str]:
        """Serialize observation for JSON transmission"""
        if obs is None:
            return None

        if isinstance(obs, np.ndarray):
            # For images, encode as base64
            if len(obs.shape) >= 2 and obs.shape[-1] in [1, 3, 4]:  # Likely an image
                if obs.dtype != np.uint8:
                    obs = (obs * 255).astype(np.uint8) if obs.max() <= 1.0 else obs.astype(np.uint8)
                return {
                    "type": "image",
                    "shape": obs.shape,
                    "data": base64.b64encode(obs.tobytes()).decode('utf-8')
                }
            else:
                # For other arrays, convert to list
                return {
                    "type": "array",
                    "shape": obs.shape,
                    "data": obs.tolist()
                }
        elif isinstance(obs, (int, float, bool)):
            return obs
        elif isinstance(obs, (list, tuple)):
            return list(obs)
        elif isinstance(obs, dict):
            return {k: self._serialize_observation(v) for k, v in obs.items()}
        else:
            return str(obs)

    def _serialize_info(self, info) -> Dict:
        """Serialize info dict for JSON transmission"""
        if info is None:
            return {}

        result = {}
        for k, v in info.items():
            if isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                result[k] = v.item()
            elif isinstance(v, (int, float, bool, str, list, dict)):
                result[k] = v
            else:
                result[k] = str(v)
        return result

    def _parse_action(self, action_payload: Dict[str, Any]) -> Any:
        """Parse action payload into Gymnasium action"""
        if self.env is None:
            raise ValueError("Environment not initialized")

        action_space = self.env.action_space

        if isinstance(action_space, Discrete):
            action = action_payload.get("action", 0)
            if not isinstance(action, int) or action < 0 or action >= action_space.n:
                raise ValueError(f"Invalid discrete action {action}, must be 0-{action_space.n-1}")
            return action

        elif isinstance(action_space, Box):
            action = action_payload.get("action", [])
            if not isinstance(action, list):
                raise ValueError("Box action must be a list")

            action = np.array(action, dtype=action_space.dtype)
            if action.shape != action_space.shape:
                raise ValueError(f"Action shape {action.shape} doesn't match space {action_space.shape}")

            return action

        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")

    def _describe_action_space(self) -> Dict[str, Any]:
        """Describe the action space"""
        if self.env is None:
            return {}

        action_space = self.env.action_space

        if isinstance(action_space, Discrete):
            return {
                "type": "discrete",
                "n": action_space.n,
                "description": f"Integer action from 0 to {action_space.n-1}"
            }
        elif isinstance(action_space, Box):
            return {
                "type": "box",
                "shape": action_space.shape,
                "low": action_space.low.tolist(),
                "high": action_space.high.tolist(),
                "dtype": str(action_space.dtype),
                "description": f"Continuous action vector of shape {action_space.shape}"
            }
        else:
            return {"type": str(type(action_space)), "description": str(action_space)}

    def _describe_observation_space(self) -> Dict[str, Any]:
        """Describe the observation space"""
        if self.env is None:
            return {}

        obs_space = self.env.observation_space

        if isinstance(obs_space, Discrete):
            return {
                "type": "discrete",
                "n": obs_space.n,
                "description": f"Integer observation from 0 to {obs_space.n-1}"
            }
        elif isinstance(obs_space, Box):
            return {
                "type": "box",
                "shape": obs_space.shape,
                "low": obs_space.low.tolist() if obs_space.low.size < 100 else "large_array",
                "high": obs_space.high.tolist() if obs_space.high.size < 100 else "large_array",
                "dtype": str(obs_space.dtype),
                "description": f"Continuous observation vector of shape {obs_space.shape}"
            }
        else:
            return {"type": str(type(obs_space)), "description": str(obs_space)}

    def close(self):
        """Close the environment"""
        if self.env is not None:
            self.env.close()
            self.env = None

    def __del__(self):
        """Cleanup on deletion"""
        self.close()

    @classmethod
    def get_tool_definitions(cls) -> List[GameToolDefinition]:
        """Base tool definitions for Gymnasium games"""
        return [
            GameToolDefinition(
                name="gym_step",
                description="Take a step in the Gymnasium environment",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"},
                        "action": {
                            "description": "Action to take (integer for discrete, array for continuous)",
                            "oneOf": [
                                {"type": "integer"},
                                {"type": "array", "items": {"type": "number"}}
                            ]
                        },
                        "player": {"type": "string", "description": "Player making the move"}
                    },
                    "required": ["game_id", "action", "player"]
                }
            ),
            GameToolDefinition(
                name="gym_reset",
                description="Reset the Gymnasium environment",
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "description": "Game instance ID"}
                    },
                    "required": ["game_id"]
                }
            )
        ]

    @classmethod
    def get_game_info(cls) -> Dict[str, Any]:
        """Base game info for Gymnasium games"""
        return {
            "type": "gymnasium",
            "name": "Gymnasium Environment",
            "description": "Base class for Gymnasium environments",
            "min_players": 1,
            "max_players": 1,
            "custom_tools": ["gym_step", "gym_reset"]
        }