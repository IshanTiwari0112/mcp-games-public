"""Base class for PettingZoo environment games"""

import json
import base64
from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    import pettingzoo
    from pettingzoo.utils.env import ParallelEnv, AECEnv
    PETTINGZOO_AVAILABLE = True
except ImportError:
    PETTINGZOO_AVAILABLE = False
    ParallelEnv = object
    AECEnv = object

from .types import Game, GameAction, GameResult, GameState, GameStatus, GameToolDefinition


class PettingZooGame(Game):
    """Base class for wrapping PettingZoo environments as MCP games"""

    def __init__(self, game_id: str, env_name: str, **env_kwargs):
        super().__init__(game_id, env_name)
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.env = None
        self.agents = []
        self.observations = {}
        self.infos = {}
        self.rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.current_agent_index = 0
        self.episode_steps = 0
        self.episode_rewards = {}
        
    def _create_env(self):
        """Create the PettingZoo environment"""
        if not PETTINGZOO_AVAILABLE:
            raise ImportError("PettingZoo is not available")
            
        if self.env is None:
            try:
                # Import the specific environment
                if "classic" in self.env_name:
                    from pettingzoo.classic import chess_v6, connect_four_v3, go_v5, tictactoe_v3
                    env_map = {
                        "chess_v6": chess_v6,
                        "connect_four_v3": connect_four_v3, 
                        "go_v5": go_v5,
                        "tictactoe_v3": tictactoe_v3
                    }
                    if self.env_name in env_map:
                        self.env = env_map[self.env_name].env(**self.env_kwargs)
                    else:
                        raise ValueError(f"Unknown classic environment: {self.env_name}")
                else:
                    raise ValueError(f"Environment category not supported: {self.env_name}")
                
                # Reset first to initialize agents
                self.env.reset()
                self.agents = self.env.agents.copy()
                
            except Exception as e:
                raise RuntimeError(f"Failed to create PettingZoo environment {self.env_name}: {str(e)}")

    def start(self, players: List[str]) -> GameResult:
        """Start the PettingZoo environment"""
        try:
            self._create_env()
            
            # PettingZoo environments have predefined agents
            if len(players) != len(self.agents):
                return GameResult(
                    success=False,
                    error=f"{self.env_name} requires exactly {len(self.agents)} players, got {len(players)}"
                )

            # Reset the environment (already done in _create_env, but ensure clean state)
            if hasattr(self.env, 'reset'):
                self.env.reset()
            
            # Map players to agents
            self.player_to_agent = {players[i]: self.agents[i] for i in range(len(players))}
            self.agent_to_player = {self.agents[i]: players[i] for i in range(len(players))}
            
            self.state.players = players
            self.state.status = GameStatus.ACTIVE
            self.state.moves = 0
            self.episode_steps = 0
            self.episode_rewards = {player: 0.0 for player in players}
            
            # Get initial observations
            self._update_observations()
            
            # Set current player
            if hasattr(self.env, 'agent_selection'):
                current_agent = self.env.agent_selection
                self.state.current_player = self.agent_to_player.get(current_agent, players[0])
            else:
                self.state.current_player = players[0]

            # Store the initial state
            self.state.state = {
                "observations": self._serialize_observations(),
                "infos": self._serialize_infos(),
                "episode_rewards": self.episode_rewards,
                "episode_steps": self.episode_steps,
                "current_agent": getattr(self.env, 'agent_selection', self.agents[0]),
                "action_space": self._describe_action_space(),
                "observation_space": self._describe_observation_space(),
            }

            return GameResult(success=True, new_state=self.state)

        except Exception as e:
            return GameResult(success=False, error=f"Failed to start environment: {str(e)}")

    def make_move(self, action: GameAction) -> GameResult:
        """Execute an action in the PettingZoo environment"""
        if self.state.status != GameStatus.ACTIVE:
            return GameResult(success=False, error="Game is not active")

        if action.player != self.state.current_player:
            return GameResult(success=False, error=f"Not {action.player}'s turn (current: {self.state.current_player})")

        # Accept both "pz_step" and direct action types
        if action.type not in ["pz_step", "step"]:
            return GameResult(success=False, error=f"Invalid action type: {action.type}")

        try:
            # Get the agent for this player
            agent = self.player_to_agent[action.player]
            
            # Parse the action
            pz_action = self._parse_action(action.payload, agent)

            # Execute the action
            self.env.step(pz_action)
            
            # Update observations and state
            self._update_observations()
            
            # Update game state
            self.episode_steps += 1
            self.state.moves += 1

            # Update episode rewards
            for agent, reward in self.rewards.items():
                if agent in self.agent_to_player:
                    player = self.agent_to_player[agent]
                    self.episode_rewards[player] = self.episode_rewards.get(player, 0) + reward

            # Update state
            self.state.state.update({
                "observations": self._serialize_observations(),
                "infos": self._serialize_infos(),
                "episode_rewards": self.episode_rewards,
                "episode_steps": self.episode_steps,
                "last_rewards": {self.agent_to_player.get(a, a): r for a, r in self.rewards.items()},
                "terminations": {self.agent_to_player.get(a, a): t for a, t in self.terminations.items()},
                "truncations": {self.agent_to_player.get(a, a): t for a, t in self.truncations.items()},
                "current_agent": getattr(self.env, 'agent_selection', None),
            })

            # Update current player
            if hasattr(self.env, 'agent_selection') and self.env.agent_selection:
                self.state.current_player = self.agent_to_player.get(self.env.agent_selection)
            
            # Check if game is done
            if all(self.terminations.values()) or all(self.truncations.values()):
                self.state.status = GameStatus.COMPLETED
                
                # Determine winner based on rewards
                max_reward = max(self.episode_rewards.values())
                winners = [player for player, reward in self.episode_rewards.items() if reward == max_reward]
                if len(winners) == 1:
                    self.state.winner = winners[0]
                    self.state.metadata["result"] = f"{winners[0]} wins!"
                else:
                    self.state.metadata["result"] = f"Draw between: {', '.join(winners)}"

            return GameResult(success=True, new_state=self.state)

        except Exception as e:
            return GameResult(success=False, error=f"Action failed: {str(e)}")

    def _update_observations(self):
        """Update observations, rewards, etc. from environment"""
        if hasattr(self.env, 'observe'):
            # AEC environment
            current_agent = getattr(self.env, 'agent_selection', None)
            if current_agent:
                self.observations[current_agent] = self.env.observe(current_agent)
                
        # Update rewards, terminations, truncations
        self.rewards = getattr(self.env, 'rewards', {})
        self.terminations = getattr(self.env, 'terminations', {})
        self.truncations = getattr(self.env, 'truncations', {})
        self.infos = getattr(self.env, 'infos', {})

    def get_state(self) -> GameState:
        """Get current game state"""
        return self.state

    def reset(self) -> GameResult:
        """Reset the environment"""
        try:
            if self.env is not None:
                self.env.reset()
                self._update_observations()

            self.state.status = GameStatus.WAITING
            self.state.moves = 0
            self.episode_steps = 0
            self.episode_rewards = {player: 0.0 for player in self.state.players}

            self.state.state = {
                "observations": self._serialize_observations(),
                "infos": self._serialize_infos(),
                "episode_rewards": self.episode_rewards,
                "episode_steps": 0,
            }

            return GameResult(success=True, new_state=self.state)

        except Exception as e:
            return GameResult(success=False, error=f"Reset failed: {str(e)}")

    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Get available actions based on current agent's action space"""
        if self.env is None:
            return []

        actions = []
        current_agent = getattr(self.env, 'agent_selection', None)
        if not current_agent:
            return []
            
        action_space = self.env.action_space(current_agent)
        
        if hasattr(action_space, 'n'):  # Discrete space
            for i in range(action_space.n):
                actions.append({
                    "type": "discrete",
                    "value": i,
                    "description": f"Action {i}"
                })
        elif hasattr(action_space, 'shape'):  # Box space
            actions.append({
                "type": "continuous",
                "shape": action_space.shape,
                "low": action_space.low.tolist() if hasattr(action_space, 'low') else None,
                "high": action_space.high.tolist() if hasattr(action_space, 'high') else None,
                "description": f"Continuous action in space {action_space.shape}"
            })

        return actions

    def _serialize_observations(self) -> Dict[str, Any]:
        """Serialize observations for JSON transmission"""
        result = {}
        for agent, obs in self.observations.items():
            player = self.agent_to_player.get(agent, agent)
            result[player] = self._serialize_single_observation(obs)
        return result
    
    def _serialize_single_observation(self, obs) -> Union[List, Dict, str]:
        """Serialize a single observation"""
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
            return {k: self._serialize_single_observation(v) for k, v in obs.items()}
        else:
            return str(obs)

    def _serialize_infos(self) -> Dict[str, Any]:
        """Serialize info dicts for JSON transmission"""
        result = {}
        for agent, info in self.infos.items():
            player = self.agent_to_player.get(agent, agent)
            if info is None:
                result[player] = {}
                continue
                
            player_info = {}
            for k, v in info.items():
                if isinstance(v, np.ndarray):
                    player_info[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    player_info[k] = v.item()
                elif isinstance(v, (int, float, bool, str, list, dict)):
                    player_info[k] = v
                else:
                    player_info[k] = str(v)
            result[player] = player_info
        return result

    def _parse_action(self, action_payload: Dict[str, Any], agent: str) -> Any:
        """Parse action payload into PettingZoo action"""
        if self.env is None:
            raise ValueError("Environment not initialized")

        action_space = self.env.action_space(agent)

        if hasattr(action_space, 'n'):  # Discrete space
            action = action_payload.get("action", 0)
            if not isinstance(action, int) or action < 0 or action >= action_space.n:
                raise ValueError(f"Invalid discrete action {action}, must be 0-{action_space.n-1}")
            return action

        elif hasattr(action_space, 'shape'):  # Box space
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
        """Describe the action spaces for all agents"""
        if self.env is None:
            return {}

        result = {}
        for agent in self.agents:
            player = self.agent_to_player.get(agent, agent)
            action_space = self.env.action_space(agent)

            if hasattr(action_space, 'n'):  # Discrete
                result[player] = {
                    "type": "discrete",
                    "n": action_space.n,
                    "description": f"Integer action from 0 to {action_space.n-1}"
                }
            elif hasattr(action_space, 'shape'):  # Box
                result[player] = {
                    "type": "box",
                    "shape": action_space.shape,
                    "low": action_space.low.tolist() if hasattr(action_space, 'low') else None,
                    "high": action_space.high.tolist() if hasattr(action_space, 'high') else None,
                    "dtype": str(action_space.dtype) if hasattr(action_space, 'dtype') else None,
                    "description": f"Continuous action vector of shape {action_space.shape}"
                }
            else:
                result[player] = {"type": str(type(action_space)), "description": str(action_space)}
                
        return result

    def _describe_observation_space(self) -> Dict[str, Any]:
        """Describe the observation spaces for all agents"""
        if self.env is None:
            return {}

        result = {}
        for agent in self.agents:
            player = self.agent_to_player.get(agent, agent)
            obs_space = self.env.observation_space(agent)

            if hasattr(obs_space, 'n'):  # Discrete
                result[player] = {
                    "type": "discrete",
                    "n": obs_space.n,
                    "description": f"Integer observation from 0 to {obs_space.n-1}"
                }
            elif hasattr(obs_space, 'shape'):  # Box
                result[player] = {
                    "type": "box",
                    "shape": obs_space.shape,
                    "low": obs_space.low.tolist() if hasattr(obs_space, 'low') and obs_space.low.size < 100 else "large_array",
                    "high": obs_space.high.tolist() if hasattr(obs_space, 'high') and obs_space.high.size < 100 else "large_array",
                    "dtype": str(obs_space.dtype) if hasattr(obs_space, 'dtype') else None,
                    "description": f"Observation vector of shape {obs_space.shape}"
                }
            else:
                result[player] = {"type": str(type(obs_space)), "description": str(obs_space)}
                
        return result

    def close(self):
        """Close the environment"""
        if self.env is not None:
            if hasattr(self.env, 'close'):
                self.env.close()
            self.env = None

    def __del__(self):
        """Cleanup on deletion"""
        self.close()

    @classmethod
    def get_tool_definitions(cls) -> List[GameToolDefinition]:
        """Base tool definitions for PettingZoo games"""
        return [
            GameToolDefinition(
                name="pz_step",
                description="Take a step in the PettingZoo environment",
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
            )
        ]

    @classmethod
    def get_game_info(cls) -> Dict[str, Any]:
        """Base game info for PettingZoo games"""
        return {
            "type": "pettingzoo",
            "name": "PettingZoo Environment",
            "description": "Base class for PettingZoo multi-agent environments",
            "min_players": 2,
            "max_players": 2,
            "custom_tools": ["pz_step"]
        }
