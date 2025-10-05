from core.interfaces import TokenEncoder, Transition

# Token grammar examples:
#   Global specials: <SEP>, <P0>, <P1>, <START>, <END>
#   Kuhn actions: <CHECK>, <BET>, <CALL>, <FOLD>
#   Leduc adds streets: <PRE>, <FLOP>, <TURN> (Leduc actually: pre + public card rounds)
# You can materialize these to actual tokenizer vocab later.

class KuhnTokenEncoder(TokenEncoder):
    env_name = "kuhn"
    sep = " <SEP> "
    ACTION_TOKS = {0:"<PASS>", 1:"<BET>", 2:"<CALL>", 3:"<FOLD>"}  # adjust to actual mapping

    def encode(self, traj: Transition) -> str:
        pieces = ["<START>", f"<P{traj.player}>"]
        for i, a in enumerate(traj.history):
            pieces.append(self.ACTION_TOKS.get(a, f"<A{a}>"))
        return self.sep.join(pieces + ["<END>"])

class LeducTokenEncoder(TokenEncoder):
    env_name = "leduc"
    sep = " <SEP> "
    ACTION_TOKS = {0:"<CHECK>", 1:"<BET>", 2:"<CALL>", 3:"<FOLD>", 4:"<RAISE>"}  # example
    # Optional: inject street separators by peeking into traj.info()
    def encode(self, traj: Transition) -> str:
        pieces = ["<START>", f"<P{traj.player}>", "<PRE>"]
        # TODO: insert <PUBLIC_X> street markers when public card changes
        for a in traj.history:
            pieces.append(self.ACTION_TOKS.get(a, f"<A{a}>"))
        pieces.append("<END>")
        return self.sep.join(pieces)
