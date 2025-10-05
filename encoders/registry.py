from .dense_features import KuhnDenseEncoder, LeducDenseEncoder
from .longformer_tokens import KuhnTokenEncoder, LeducTokenEncoder

DENSE_ENCODERS = {
    "kuhn": KuhnDenseEncoder(),
    "leduc": LeducDenseEncoder(),
}
TOKEN_ENCODERS = {
    "kuhn": KuhnTokenEncoder(),
    "leduc": LeducTokenEncoder(),
}
