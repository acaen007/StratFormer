from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Dict, List, Tuple, Any, Optional

Action = int
PlayerID = int

@dataclass
class Transition:
    obs: Any                # env-specific obs for current player
    info: Dict[str, Any]    # extra: public cards, pot, bet size, etc.
    legal_actions: List[Action]
    history: List[Action]   # full action history from root (env encoding)
    player: PlayerID
    action: Optional[Action] = None
    reward: Optional[float] = None
    terminal: bool = False

class Env(Protocol):
    name: str
    num_players: int
    action_dim: int

    def reset(self) -> None: ...
    def current_player(self) -> PlayerID: ...
    def legal_actions(self) -> List[Action]: ...
    def observation(self, player: PlayerID) -> Any: ...
    def info(self) -> Dict[str, Any]: ...
    def history(self) -> List[Action]: ...
    def is_terminal(self) -> bool: ...
    def step(self, a: Action) -> None: ...
    def returns(self) -> List[float]: ...   # at terminal

class Policy(Protocol):
    name: str
    env_name: str
    def action_probs(self, env: Env, player: PlayerID) -> Dict[Action, float]: ...
    def act(self, env: Env, player: PlayerID, rng=None) -> Action: ...

class DenseEncoder(Protocol):
    """For Ridge: history → fixed-length numeric vector."""
    env_name: str
    def encode(self, traj: Transition) -> Tuple[List[float], List[int]]:
        """Returns (x_vector, legal_actions)"""

class TokenEncoder(Protocol):
    """For Longformer: history → token string or ids."""
    env_name: str
    sep: str
    def encode(self, traj: Transition) -> str:
        """Returns a single sequence string ready for tokenizer."""
