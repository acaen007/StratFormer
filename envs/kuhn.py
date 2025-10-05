from .openspiel_base import OpenSpielStateEnv

class KuhnEnv(OpenSpielStateEnv):
    def __init__(self):
        super().__init__("kuhn_poker")
        self.name = "kuhn"
