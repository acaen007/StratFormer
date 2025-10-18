import pyspiel

class OpenSpielStateEnv:
    """Minimal, deterministic wrapper around pyspiel State."""
    def __init__(self, game_name: str):
        self.game_name = game_name
        self._game = pyspiel.load_game(game_name)
        self._state = None
        # make these visible for interfaces
        self.name = game_name
        self.num_players = self._game.num_players()
        self.action_dim = self._game.num_distinct_actions()

    def reset(self):
        self._state = self._game.new_initial_state()

    def current_player(self):
        return self._state.current_player() if self._state is not None else 0

    def legal_actions(self):
        return self._state.legal_actions() if self._state is not None else []

    def observation(self, player):
        # For imperfect-info games: use information state string for player.
        return self._state.information_state_string(player) if self._state is not None else None

    def info(self):
        # Optionally expose public info; keep generic.
        return {"public_obs": self._state.observation_tensor() if self._state is not None else None
                if self._game.get_type().provides_information_state_tensor
                else None}

    def history(self):
        return self._state.history() if self._state is not None else []

    def is_terminal(self):
        return self._state.is_terminal() if self._state is not None else False

    def step(self, a: int):
        self._state.apply_action(a) if self._state is not None else None    

    def returns(self):
        return self._state.returns() if self._state is not None else [0.0] * self.num_players
