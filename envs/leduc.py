from .openspiel_base import OpenSpielStateEnv

class LeducEnv(OpenSpielStateEnv):
    def __init__(self):
        super().__init__("leduc_poker")
        self.name = "leduc"
