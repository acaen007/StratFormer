from typing import Dict, List, cast
from envs.kuhn import KuhnEnv
from envs.leduc import LeducEnv
from encoders.registry import DENSE_ENCODERS, TOKEN_ENCODERS
from policies.registry import load_baselines
from core.interfaces import Policy

def load_env(name: str):
    name = name.lower()
    if name == "kuhn":  return KuhnEnv()
    if name == "leduc": return LeducEnv()
    raise ValueError(f"Unknown env {name}")

def load_encoders(env_name: str):
    return DENSE_ENCODERS[env_name], TOKEN_ENCODERS[env_name]

def load_policy_pool(env_name: str, include: List[str] | None = None) -> Dict[str, Policy]:
    pool = load_baselines(env_name)
    if include:
        pool = {k: v for k, v in pool.items() if k in include}
    return cast(Dict[str, Policy], pool)
